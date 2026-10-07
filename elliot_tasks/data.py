"""Where Major TOM ELLIOT-Pretrain and ELLIOT-X-EXT live, and one tile read from both.

Both datasets come one TACO collection per part (`monotemporal`, `monthly`, `burst`),
and they are ROW-ALIGNED: sample `i` of ELLIOT-X-EXT's part extends sample `i` of
ELLIOT-Pretrain's, and both carry the same `majortom:code_10km`. Everything is read
through `taco.ml.Dataset`; no path to a payload is built by hand.

A root holds the three part folders. Each part folder is either a TACO FOLDER
container (`COLLECTION.json`, `METADATA/`, `DATA/`) or a set of TACO zip parts with
their `.tacocat/` catalog. The root itself is one of:

  /local/dir                     already on disk
  hf://<org>/<repo>[/<subdir>]   a Hugging Face dataset, e.g. hf://isp-uv-es/elliot-x-ext
  https://...                    a plain HTTP mirror of a FOLDER layout, e.g.
                                 https://data.source.coop/major-tom/elliot-pretrain

Remote roots are mirrored lazily into a cache directory: a part's metadata when the
part is opened, and only the files (FOLDER) or the one zip part (TACOCAT) holding a
sample when that sample is read. Nothing close to the multi-TB whole is fetched for
a handful of tiles.

A root can also take a METADATA OVERLAY: a directory or a zip file (local, or
`hf://<org>/<repo>/<path>`) holding `<part>/COLLECTION.json` and `<part>/METADATA/`
that replace the root's own. This is how ELLIOT-Pretrain's corrected,
contract-bearing metadata is used over pixels read from its release; when a release
has no `ml:contract`, `ELLIOT_METADATA_OVERLAY` is used automatically. See the README.

Roots come from `configure(...)`, or else from the environment variables
`ELLIOT_ROOT`, `ELLIOT_X_EXT_ROOT` and, optionally, `ELLIOT_METADATA`.
"""
from __future__ import annotations

import functools
import hashlib
import io
import json
import os
import shutil
import urllib.request
import warnings
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

PARTS = ('monotemporal', 'monthly', 'burst')

#: The modality keys used everywhere in this package, and the slot that holds each.
#: `elliot` slots come from ELLIOT-Pretrain, `ext` slots from ELLIOT-X-EXT.
SLOTS = {
    's2': ('elliot', 's2'),
    'l8': ('elliot', 'l8'),
    'dem': ('elliot', 'dem'),
    'lc': ('elliot', 'land_cover'),
    's1': ('ext', 's1'),
    'cloud_mask_s2': ('ext', 'cloud_mask_s2'),
    'cloud_mask_l8': ('ext', 'cloud_mask_l8'),
}

#: The modalities a fact or a task can require. `series` is a pseudo-modality: a
#: stack of more than one acquisition (see `tasks.TileFacts.available`).
MODALITIES = ('s2', 'l8', 's1', 'dem', 'lc')

TILE_PX = 1056
CATALOG = '.tacocat'


# --------------------------------------------------------------------------------
# Roots.
# --------------------------------------------------------------------------------
def _cache_base(cache_dir: str | None) -> Path:
    return Path(cache_dir or os.environ.get(
        'ELLIOT_TASKS_CACHE', Path.home() / '.cache' / 'elliot-tasks'))


def _metadata_files(collection: dict) -> list[str]:
    """`METADATA/` file names of a FOLDER part, from its declared levels."""
    levels = collection.get('taco:metadata') or {}
    names = {'sample.parquet'} | {f'{lvl.replace("/", "__")}.parquet' for lvl in levels}
    return sorted(names)


def _fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + f'.{os.getpid()}.part')   # safe across workers
    req = urllib.request.Request(url, headers={'User-Agent': 'majortom-elliot-tasks'})
    with urllib.request.urlopen(req) as r, open(tmp, 'wb') as fh:
        shutil.copyfileobj(r, fh)
    tmp.replace(dest)


def _ready(folder: Path) -> bool:
    """Is a part's metadata complete on disk: a catalog, or COLLECTION.json + METADATA?"""
    return (folder / CATALOG / 'sample.parquet').is_file() or (
        (folder / 'COLLECTION.json').is_file()
        and (folder / 'METADATA' / 'sample.parquet').is_file())


@functools.lru_cache(maxsize=16)
def _partitions(catalog: str) -> list[str]:
    """The zip part holding each row of a TACOCAT, in row order."""
    import pyarrow.parquet as pq
    t = pq.read_table(Path(catalog) / 'sample.parquet', columns=['internal:source_file'])
    return t.column('internal:source_file').to_pylist()


#: The corrected ELLIOT-Pretrain metadata, published beside ELLIOT-X-EXT. It is used
#: automatically when an ELLIOT-Pretrain root's own COLLECTION.json has no
#: `ml:contract`, which is the case for the current Source Cooperative release.
ELLIOT_METADATA_OVERLAY = 'hf://isp-uv-es/elliot-x-ext/elliot-pretrain-metadata-overlay.zip'


def _split_hf(spec: str) -> tuple[str, str]:
    """`hf://<org>/<repo>[/<path>]` -> (`<org>/<repo>`, `<path>`)."""
    bits = spec.removeprefix('hf://').strip('/').split('/')
    if len(bits) < 2:
        raise ValueError(f'{spec!r}: expected hf://<org>/<repo>[/<path>]')
    return '/'.join(bits[:2]), '/'.join(bits[2:])


_PART_DIRS: dict = {}


@dataclass(frozen=True)
class Source:
    """One dataset root (see the module docstring), with an optional metadata overlay.

    `metadata` is an overlay chosen by the caller. `default_metadata` is used only
    when the root's own metadata has no `ml:contract`, and says so when it is.
    """
    spec: str
    cache_dir: str | None = None
    metadata: str | None = None
    default_metadata: str | None = None

    @property
    def kind(self) -> str:
        if self.spec.startswith('hf://'):
            return 'hf'
        if self.spec.startswith(('http://', 'https://')):
            return 'http'
        return 'local'

    # -- where things go --------------------------------------------------------
    def _hf(self) -> tuple[str, str]:
        repo, sub = _split_hf(self.spec)
        return repo, (sub + '/' if sub else '')

    def _hf_local(self, repo: str) -> Path:
        return _cache_base(self.cache_dir) / 'hf' / repo.replace('/', '__')

    def _mirror(self, overlay: str | None) -> Path:
        """Local folder holding the parts: the root itself, or its cache mirror."""
        if self.kind == 'local' and not overlay:
            return Path(self.spec)
        slug = hashlib.sha1(f'{self.spec}|{overlay}'.encode()).hexdigest()[:10]
        name = self.spec.rstrip('/').split('/')[-1] or 'root'
        return _cache_base(self.cache_dir) / f'{name}-{slug}'

    def _hf_download(self, spec: str, patterns: list[str]) -> Path:
        from huggingface_hub import snapshot_download
        repo, _ = _split_hf(spec)
        local = self._hf_local(repo)
        snapshot_download(repo_id=repo, repo_type='dataset', allow_patterns=patterns,
                          local_dir=str(local))
        return local

    # -- metadata -------------------------------------------------------------
    def _release_collection(self, part: str) -> dict | None:
        """The root's own COLLECTION.json for one part, or None if it has none."""
        if self.kind == 'local':
            for f in (Path(self.spec) / part / 'COLLECTION.json',
                      Path(self.spec) / part / CATALOG / 'COLLECTION.json'):
                if f.is_file():
                    return json.loads(f.read_text())
            return None
        if self.kind == 'hf':
            _, sub = self._hf()
            local = self._hf_download(self.spec, [f'{sub}{part}/COLLECTION.json',
                                                  f'{sub}{part}/{CATALOG}/COLLECTION.json'])
            for f in (local / sub / part / 'COLLECTION.json',
                      local / sub / part / CATALOG / 'COLLECTION.json'):
                if f.is_file():
                    return json.loads(f.read_text())
            return None
        dest = self._mirror(None) / '_release' / part / 'COLLECTION.json'
        if not dest.is_file():
            _fetch(f'{self.spec.rstrip("/")}/{part}/COLLECTION.json', dest)
        return json.loads(dest.read_text())

    def overlay_for(self, part: str) -> str | None:
        """The metadata overlay this part is read with, if any."""
        if self.metadata:
            return self.metadata
        if not self.default_metadata:
            return None
        coll = self._release_collection(part)
        if coll is None or 'ml:contract' in coll:
            return None
        warnings.warn(
            f'{self.spec} ({part}): this release of Major TOM ELLIOT-Pretrain has no '
            f'ml:contract and carries the old frame metadata, so its corrected metadata '
            f'is used instead, from {self.default_metadata}. Pixels are still read from '
            f'{self.spec}. Pass elliot_metadata=... to use another overlay.',
            stacklevel=4)
        return self.default_metadata

    def _overlay(self, part: str, folder: Path, src: str) -> None:
        """Put an overlay's `<part>/COLLECTION.json` and `<part>/METADATA/` into `folder`.

        The overlay is a directory or a zip file holding those, locally or as
        `hf://<org>/<repo>/<path>`.
        """
        if src.endswith('.zip'):
            if src.startswith('hf://'):
                from huggingface_hub import hf_hub_download
                repo, path = _split_hf(src)
                src = hf_hub_download(repo_id=repo, repo_type='dataset', filename=path,
                                      local_dir=str(self._hf_local(repo)))
            with zipfile.ZipFile(src) as z:
                names = [n for n in z.namelist() if n.startswith(f'{part}/')
                         and not n.endswith('/')]
                if f'{part}/COLLECTION.json' not in names:
                    raise FileNotFoundError(f'metadata overlay {src}: no {part}/COLLECTION.json')
                for n in names:
                    dest = folder / n.removeprefix(f'{part}/')
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with z.open(n) as r, open(dest, 'wb') as w:
                        shutil.copyfileobj(r, w)
            return
        if src.startswith('hf://'):
            _, sub = _split_hf(src)
            sub = sub + '/' if sub else ''
            meta = self._hf_download(src, [f'{sub}{part}/COLLECTION.json',
                                           f'{sub}{part}/METADATA/*']) / sub / part
        else:
            meta = Path(src) / part
        if not (meta / 'COLLECTION.json').is_file():
            raise FileNotFoundError(f'metadata overlay {src}: no {part}/COLLECTION.json')
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copytree(meta / 'METADATA', folder / 'METADATA', dirs_exist_ok=True)
        shutil.copy2(meta / 'COLLECTION.json', folder / 'COLLECTION.json')

    def part_dir(self, part: str) -> Path:
        """The local folder of one part, with its metadata present. Resolved once
        per process, so the overlay check (and its warning) happens once."""
        key = (self, part)
        if key not in _PART_DIRS:
            _PART_DIRS[key] = self._part_dir(part)
        return _PART_DIRS[key]

    def _part_dir(self, part: str) -> Path:
        overlay = self.overlay_for(part)
        if overlay:
            folder = self._mirror(overlay) / part
            if not _ready(folder):
                self._overlay(part, folder, overlay)
            if self.kind == 'local' and not (folder / 'DATA').exists():
                # Pixels stay where they are.
                (folder / 'DATA').symlink_to((Path(self.spec) / part / 'DATA').resolve())
            return folder
        if self.kind == 'hf':
            repo, sub = self._hf()
            folder = self._hf_local(repo) / sub / part
        else:
            folder = self._mirror(None) / part
        if self.kind == 'local' or _ready(folder):
            return folder
        if self.kind == 'hf':
            repo, sub = self._hf()
            self._hf_download(self.spec, [f'{sub}{part}/{CATALOG}/*',
                                          f'{sub}{part}/COLLECTION.json',
                                          f'{sub}{part}/METADATA/*'])
        elif self.kind == 'http':
            base = self.spec.rstrip('/') + f'/{part}'
            coll = self._release_collection(part)
            if 'ml:contract' not in coll:
                raise ValueError(_NO_CONTRACT.format(what=self.spec, where=base))
            for name in _metadata_files(coll):
                _fetch(f'{base}/METADATA/{name}', folder / 'METADATA' / name)
            shutil.copy2(self._mirror(None) / '_release' / part / 'COLLECTION.json',
                         folder / 'COLLECTION.json')
        return folder

    def ensure_sample(self, part: str, index: int, paths: list[str]) -> None:
        """Make one sample's payloads local. `paths` are its relative paths below
        `DATA/` (used by FOLDER layouts); a TACOCAT fetches the zip part instead."""
        if self.kind == 'local':
            return
        folder = self.part_dir(part)
        if (folder / CATALOG).is_dir():
            zipname = _partitions(str(folder / CATALOG))[index]
            if not (folder / zipname).is_file():
                _, sub = self._hf()
                self._hf_download(self.spec, [f'{sub}{part}/{zipname}'])
            return
        missing = [p for p in paths if not (folder / 'DATA' / p).is_file()]
        if not missing:
            return
        if self.kind == 'hf':
            _, sub = self._hf()
            self._hf_download(self.spec, [f'{sub}{part}/DATA/{p}' for p in missing])
        else:
            base = self.spec.rstrip('/') + f'/{part}/DATA'
            for p in missing:
                _fetch(f'{base}/{p}', folder / 'DATA' / p)


@dataclass(frozen=True)
class Roots:
    elliot: Source
    ext: Source


_ROOTS: Roots | None = None


def configure(elliot: str | None = None, ext: str | None = None, *,
              elliot_metadata: str | None = None,
              cache_dir: str | None = None) -> Roots:
    """Set the two dataset roots for this process. Returns them.

    `elliot_metadata` is a metadata overlay for ELLIOT-Pretrain (see the module
    docstring), defaulting to the `ELLIOT_METADATA` environment variable. Without
    one, a release whose metadata has no `ml:contract` -- the current Source
    Cooperative release -- is read with `ELLIOT_METADATA_OVERLAY`, with a warning.
    """
    global _ROOTS
    elliot = elliot or os.environ.get('ELLIOT_ROOT')
    ext = ext or os.environ.get('ELLIOT_X_EXT_ROOT')
    elliot_metadata = elliot_metadata or os.environ.get('ELLIOT_METADATA') or None
    if not elliot or not ext:
        raise RuntimeError('set both roots: elliot_tasks.configure(elliot=..., ext=...) '
                           'or the ELLIOT_ROOT and ELLIOT_X_EXT_ROOT variables')
    _ROOTS = Roots(Source(str(elliot), cache_dir, elliot_metadata, ELLIOT_METADATA_OVERLAY),
                   Source(str(ext), cache_dir))
    _open.cache_clear()
    return _ROOTS


def roots() -> Roots:
    return _ROOTS or configure()


_NO_CONTRACT = ('{what} at {where} has no ml:contract in its COLLECTION.json. The '
                'release metadata of Major TOM ELLIOT-Pretrain predates it: pass a '
                'metadata overlay (configure(elliot_metadata=...) or ELLIOT_METADATA).')


def _open_dataset(folder: Path, what: str):
    import taco.ml
    try:
        return taco.ml.Dataset(str(folder))
    except ValueError as err:
        if 'ml:contract' in str(err):
            raise ValueError(_NO_CONTRACT.format(what=what, where=folder)) from err
        raise


def _read_location(loc: str) -> bytes:
    """Bytes of a payload located by the TACO reader: a file, or a byte range of a
    zip part written as GDAL's `/vsisubfile/<offset>_<size>,<file>`."""
    if loc.startswith('/vsisubfile/'):
        span, _, path = loc.removeprefix('/vsisubfile/').partition(',')
        offset, size = (int(v) for v in span.split('_'))
        with open(path, 'rb') as fh:
            fh.seek(offset)
            return fh.read(size)
    return Path(loc).read_bytes()


# --------------------------------------------------------------------------------
# Parts.
# --------------------------------------------------------------------------------
class Part:
    """One part of both datasets, opened through `taco.ml.Dataset`."""

    def __init__(self, name: str, r: Roots):
        if name not in PARTS:
            raise ValueError(f'part must be one of {PARTS}, got {name!r}')
        self.name, self.roots = name, r
        self.elliot = _open_dataset(r.elliot.part_dir(name), 'ELLIOT-Pretrain')
        self.ext = _open_dataset(r.ext.part_dir(name), 'ELLIOT-X-EXT')
        if len(self.elliot) != len(self.ext):
            raise ValueError(f'{name}: {len(self.elliot)} ELLIOT samples against '
                             f'{len(self.ext)} in the extension; they must be row-aligned')

    def __len__(self) -> int:
        return len(self.elliot)

    def __repr__(self) -> str:
        return f'Part({self.name!r}, {len(self)} tiles)'

    @functools.cached_property
    def codes(self) -> list[str]:
        """Bare cell code (no `MT10_` prefix) of every row, in row order."""
        return [c.removeprefix('MT10_')
                for c in self.elliot.table.column('majortom:code_10km').to_pylist()]

    @functools.cached_property
    def index_of(self) -> dict[str, int]:
        return {c: i for i, c in enumerate(self.codes)}

    def slot(self, key: str):
        side, name = SLOTS[key]
        ds = self.elliot if side == 'elliot' else self.ext
        return next(s for s in ds.contract.inputs if s.name == name)

    def tile(self, index: int) -> Tile:
        return Tile(self, int(index))

    @functools.cached_property
    def _vector_locations(self):
        """Where each sample's `osm.parquet` and `admin.parquet` are, from `taco.read`.

        These two GeoParquet leaves are structure leaves of the extension, not slots
        of its ML contract, so they are located through the generic reader.
        """
        t = self.ext.reader.read(files=['osm.parquet', 'admin.parquet'])
        if t.column('majortom:code_10km').to_pylist() != \
                self.ext.table.column('majortom:code_10km').to_pylist():
            raise ValueError(f'{self.name}: leaf locations are not in sample order')
        return {name: t.column(f'{name}::location').to_pylist()
                for name in ('osm.parquet', 'admin.parquet')}

    def vector_location(self, name: str, index: int) -> str | None:
        return self._vector_locations[name][index]

    def payload_paths(self, side: str, index: int) -> list[str]:
        """Every payload path of one sample below `DATA/`, from the metadata."""
        import pyarrow.compute as pc
        ds = self.elliot if side == 'elliot' else self.ext
        out = []
        for level in ds.reader.contract.levels:
            if level == 'sample':
                continue
            col = ds.level(level).column('internal:relative_path')
            hit = col.filter(pc.starts_with(col, f'{index}/'))
            out += [p for p in hit.to_pylist() if p and '.' in p.rsplit('/', 1)[-1]]
        # Files at the top of a sample (the extension's GeoParquet leaves) are in the
        # declared structure even where no metadata row lists them.
        out += [f'{index}/{leaf}' for leaf in ds.collection.get('taco:structure', [])
                if '/' not in leaf]
        return sorted(set(out))


@functools.lru_cache(maxsize=8)
def _open(name: str, r: Roots) -> Part:
    return Part(name, r)


def open_part(name: str) -> Part:
    """One part of ELLIOT + ELLIOT-X-EXT, opened once per process."""
    return _open(name, roots())


def find(cell: str, part: str | None = None) -> Tile:
    """A tile by cell code (`284D_496L` or `MT10_284D_496L`).

    A cell code is unique within a part but not across them (some cells are in
    two), so without `part` the first part that holds it wins, in `PARTS` order.
    """
    cell = cell.removeprefix('MT10_')
    for name in ([part] if part else PARTS):
        p = open_part(name)
        if cell in p.index_of:
            return p.tile(p.index_of[cell])
    raise KeyError(f'cell {cell} not found in {part or PARTS}')


# --------------------------------------------------------------------------------
# One tile.
# --------------------------------------------------------------------------------
def _frame_rows(ds, index: int, level: str, columns: list[str]) -> list[dict]:
    """The rows of one frame level for one sample, in FRAME order (by file name)."""
    if level not in ds.reader.contract.levels:
        return []
    paths = ds.lookup(index, f'{level}:internal:relative_path')
    cols = {c: ds.lookup(index, f'{level}:{c}') for c in columns
            if c in ds.level(level).column_names}
    rows = [{'path': p, **{c: v[k] for c, v in cols.items()}} for k, p in enumerate(paths)]
    return sorted(rows, key=lambda r: r['path'])


ELLIOT_S2_COLUMNS = ['s2:date', 's2:spacecraft', 's2:mean_solar_zenith',
                     's2:mean_solar_azimuth', 'ml:crs', 'ml:geotransform', 'ml:time_start']
ELLIOT_L8_COLUMNS = ['l8:date', 'l8:spacecraft', 'l8:sun_elevation']
EXT_FRAME_COLUMNS = ['ml:acquisition_index', 'ml:date', 'ml:cloud_fraction', 'ml:clear',
                     'ml:thick_cloud', 'ml:thin_cloud', 'ml:shadow',
                     'ml:era5_temperature_2m_c', 'ml:era5_dewpoint_2m_c',
                     'ml:era5_relative_humidity_pct', 'ml:era5_precipitation_mm',
                     'ml:era5_cloud_cover_pct', 'ml:era5_wind_speed_10m_kmh',
                     'ml:era5_wind_from_direction_deg']
EXT_S1_COLUMNS = ['s1:s2_frame', 's1:s2_date', 's1:datetime', 's1:platform',
                  's1:orbit_state', 's1:valid_fraction', 's1:selection']


@dataclass
class Tile:
    """One ELLIOT tile: a row of a part, read from both datasets, cached.

    Every array is read at most once per `Tile`, so the fact sheet, the series facts,
    every task family and the PyTorch loader can share one read of the cube. Arrays
    come back as stored (S2 and L8 integer counts, scale 1e-4; S1 linear power; DEM
    metres; land cover WorldCover codes; cloud masks 0 clear, 1 thick cloud, 2 thin
    cloud, 3 shadow), always with a leading frame axis for the per-acquisition
    modalities, so a monotemporal tile is a series of one.
    """
    part: Part
    index: int
    _cache: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        if not 0 <= self.index < len(self.part):
            raise IndexError(f'{self.part.name} has {len(self.part)} tiles, '
                             f'not {self.index}')
        r = self.part.roots
        if r.elliot.kind != 'local':
            r.elliot.ensure_sample(self.part.name, self.index,
                                   self.part.payload_paths('elliot', self.index))
        if r.ext.kind != 'local':
            r.ext.ensure_sample(self.part.name, self.index,
                                self.part.payload_paths('ext', self.index))

    def __repr__(self) -> str:
        return f'Tile({self.part.name!r}, {self.index}, cell={self.cell!r})'

    # -- identity and metadata ------------------------------------------------
    @property
    def cell(self) -> str:
        return self.part.codes[self.index]

    @functools.cached_property
    def meta(self) -> dict[str, Any]:
        """ELLIOT-Pretrain's sample-level row."""
        return self.part.elliot.metadata(self.index)

    @functools.cached_property
    def ext_meta(self) -> dict[str, Any]:
        """ELLIOT-X-EXT's sample-level row."""
        return self.part.ext.metadata(self.index)

    @functools.cached_property
    def s2_frames(self) -> list[dict]:
        """ELLIOT's per-acquisition S2 rows (date, sun angles, CRS, geotransform)."""
        return _frame_rows(self.part.elliot, self.index, 'children/s2', ELLIOT_S2_COLUMNS)

    @functools.cached_property
    def l8_frames(self) -> list[dict]:
        return _frame_rows(self.part.elliot, self.index, 'children/l8', ELLIOT_L8_COLUMNS)

    @functools.cached_property
    def s2_ext_frames(self) -> list[dict]:
        """X-EXT's per-acquisition S2 rows: cloud fractions and ERA5 at overpass."""
        return _frame_rows(self.part.ext, self.index, 'children/s2', EXT_FRAME_COLUMNS)

    @functools.cached_property
    def l8_ext_frames(self) -> list[dict]:
        return _frame_rows(self.part.ext, self.index, 'children/l8', EXT_FRAME_COLUMNS)

    @functools.cached_property
    def s1_frames(self) -> list[dict]:
        return _frame_rows(self.part.ext, self.index, 'children/s1', EXT_S1_COLUMNS)

    @property
    def n_frames(self) -> int:
        return len(self.s2_frames)

    @property
    def lonlat(self) -> tuple[float, float] | None:
        g = self.meta.get('ml:geometry')
        if g is None:
            return None
        from shapely import wkb
        p = wkb.loads(g).centroid
        return float(p.x), float(p.y)

    def crs(self, frame: int = 0) -> str:
        return str(self.s2_frames[frame].get('ml:crs') or '')

    def geotransform(self, frame: int = 0) -> list[float]:
        """GDAL order: x0, dx, 0, y0, 0, dy -- of the S2 raster this frame is."""
        return [float(v) for v in self.s2_frames[frame]['ml:geotransform']]

    def has(self, key: str) -> bool:
        """Is this modality actually present for this tile?

        S1 counts only where some frame holds valid pixels: a frame with no VV+VH
        pass is stored as an all-nodata raster, which is a hole, not an observation.
        """
        if key == 's1':
            return any((r.get('s1:valid_fraction') or 0) > 0 for r in self.s1_frames)
        if key == 'l8':
            return bool(self.l8_frames)
        if key == 's2':
            return bool(self.s2_frames)
        return key in SLOTS

    # -- arrays ---------------------------------------------------------------
    def read(self, key: str) -> np.ndarray | None:
        """One modality as stored; a frame axis first for per-acquisition ones."""
        if key in self._cache:
            return self._cache[key]
        side, name = SLOTS[key]
        ds = self.part.elliot if side == 'elliot' else self.part.ext
        got = ds.read(self.index, slots=[name]).get(name)
        arr = None
        if got is not None:
            arr = np.asarray(got.array)
            kind = got.slot.kind.value
            if key in ('s2', 'l8', 's1') and kind == 'raster':
                arr = arr[None]                       # (C,H,W) -> (1,C,H,W)
            elif key.startswith('cloud_mask') and kind == 'mask':
                arr = arr[None]                       # (H,W) -> (1,H,W)
            elif key == 'dem':
                arr = arr.reshape(arr.shape[-2:])
        self._cache[key] = arr
        return arr

    def frame(self, key: str, frame: int = 0) -> np.ndarray | None:
        arr = self.read(key)
        if arr is None or frame >= arr.shape[0]:
            return None
        return arr[frame]

    def scale(self, key: str) -> float:
        s = self.part.slot(key).scale_factor
        return 1.0 if s is None else float(s)

    # -- vectors --------------------------------------------------------------
    def _parquet(self, name: str):
        """One of the GeoParquet leaves, `osm.parquet` or `admin.parquet`."""
        if name in self._cache:
            return self._cache[name]
        import pyarrow.parquet as pq
        loc = self.part.vector_location(name, self.index)
        df = pq.read_table(io.BytesIO(_read_location(loc))).to_pandas() if loc else None
        self._cache[name] = df
        return df

    def osm(self):
        """Every OSM feature of the tile, geometry WKB in the tile's UTM CRS."""
        return self._parquet('osm.parquet')

    def admin(self):
        """Administrative units intersecting the tile, clipped to it."""
        return self._parquet('admin.parquet')

    def drop_arrays(self) -> None:
        """Release the cached arrays (keeps metadata)."""
        self._cache.clear()


def tile(part: str, index: int) -> Tile:
    return open_part(part).tile(index)
