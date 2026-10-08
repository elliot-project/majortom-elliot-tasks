"""Precomputed tile facts: every task and caption without reading a pixel.

Building a tile's fact sheet reads its whole cube (S2, L8, S1, DEM, land cover, cloud
masks, OpenStreetMap), about two seconds per tile. The facts store holds, for every
tile, what the task families and the captioner read: the fact sheet, the series facts
of a monthly or burst tile, and the NDVI and land-cover grids at every side a grid task
can ask for. With it, `examples_for` and `caption` give exactly what they give from
the pixels, at the cost of reading a few kilobytes.

The stored numbers are what the pixels give on the CPU that built the store. Float32
arithmetic can differ in its last digits between CPU types, so a sheet recomputed
elsewhere may differ there (and, rarely, by one pixel at a threshold); the tasks built
from either are the same in practice.

Layout: `<root>/<part>/facts-<first index>.parquet`, `SHARD` tiles per file, with
columns `index`, `cell` and `facts` (JSON, see `encode`). The JSON has no pickled
objects: arrays, geometries (WKB), sets, tuples and the two dataclasses are tagged.

    import elliot_tasks as et
    et.configure_facts('/data/elliot-facts')        # or ELLIOT_FACTS
    t = et.facts('monthly', 352)                     # a TileFacts, no pixels read
    et.examples_for(facts=t, available={'s2', 'dem'})
"""
from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import os
from enum import Enum
from pathlib import Path

import numpy as np

from . import factsheet as fsheet
from . import temporal as tmp

SHARD = 1000
GRID_SIDES = range(3, 11)          # every side `tasks._grid_side` can draw
VERSION = 1

_CLASSES = {'FactSheet': fsheet.FactSheet, 'SeriesFacts': tmp.SeriesFacts}
_ENUMS = {'SeriesKind': tmp.SeriesKind}


def _enc(o):
    if isinstance(o, np.ndarray):
        return {'__nd__': _enc(o.tolist()), 'dtype': str(o.dtype), 'shape': list(o.shape)}
    if isinstance(o, np.generic):
        return {'__np__': str(o.dtype), 'v': o.item()}
    if isinstance(o, Enum) and type(o).__name__ in _ENUMS:
        return {'__enum__': type(o).__name__, 'v': o.value}
    if dataclasses.is_dataclass(o) and type(o).__name__ in _CLASSES:
        return {'__dc__': type(o).__name__,
                'f': {f.name: _enc(getattr(o, f.name)) for f in dataclasses.fields(o)}}
    if isinstance(o, dict):
        if all(isinstance(k, str) and not k.startswith('__') for k in o):
            return {k: _enc(v) for k, v in o.items()}
        return {'__dict__': [[_enc(k), _enc(v)] for k, v in o.items()]}
    if isinstance(o, tuple):
        return {'__tuple__': [_enc(v) for v in o]}
    if isinstance(o, (set, frozenset)):
        return {'__set__': [_enc(v) for v in sorted(o, key=repr)]}
    if isinstance(o, list):
        return [_enc(v) for v in o]
    if o is None or isinstance(o, (bool, int, float, str)):
        return o
    if hasattr(o, 'wkb') and hasattr(o, 'geom_type'):
        return {'__wkb__': o.wkb_hex}
    raise TypeError(f'cannot encode {type(o).__name__}')


def _dec(o):
    if isinstance(o, list):
        return [_dec(v) for v in o]
    if not isinstance(o, dict):
        return o
    if '__nd__' in o:
        return np.array(_dec(o['__nd__']), dtype=o['dtype']).reshape(o['shape'])
    if '__np__' in o:
        return np.dtype(o['__np__']).type(o['v'])
    if '__enum__' in o:
        return _ENUMS[o['__enum__']](o['v'])
    if '__dc__' in o:
        return _CLASSES[o['__dc__']](**{k: _dec(v) for k, v in o['f'].items()})
    if '__dict__' in o:
        return {_dec(k): _dec(v) for k, v in o['__dict__']}
    if '__tuple__' in o:
        return tuple(_dec(v) for v in o['__tuple__'])
    if '__set__' in o:
        return {_dec(v) for v in o['__set__']}
    if '__wkb__' in o:
        import shapely
        return shapely.from_wkb(bytes.fromhex(o['__wkb__']))
    return {k: _dec(v) for k, v in o.items()}


def encode(record: dict) -> str:
    return json.dumps(_enc(record), separators=(',', ':'), allow_nan=True)


def decode(text: str) -> dict:
    return _dec(json.loads(text))


def compute(tile) -> dict:
    """Everything the families and the captioner read from this tile's pixels."""
    from . import tasks as T
    t = T.tile_facts(tile)
    return {'version': VERSION, 'part': tile.part.name, 'index': tile.index,
            'cell': t.cell, 'sheet': t.sheet, 'series': t.series,
            'grids': {'ndvi': T.ndvi_grids(t, GRID_SIDES),
                      'cover': T.cover_grids(t, GRID_SIDES)}}


# --------------------------------------------------------------------------------
# Reading.
# --------------------------------------------------------------------------------
_ROOT: str | None = None


def configure_facts(root: str | None) -> None:
    """Point the package at a facts store: a local directory or an HTTP URL."""
    global _ROOT
    _ROOT = root
    _shard.cache_clear()


def facts_root() -> str | None:
    """The configured store, else `ELLIOT_FACTS`, else `facts/` under the merged
    release's root when that root is configured and holds one."""
    root = _ROOT or os.environ.get('ELLIOT_FACTS')
    if root:
        return root
    from . import data
    r = data._ROOTS
    if r is not None and r.ext is None:
        spec = r.elliot.spec.rstrip('/')
        if spec.startswith(('http://', 'https://')) or (Path(spec) / 'facts').is_dir():
            return f'{spec}/facts'
    return None


def shard_name(part: str, index: int) -> str:
    return f'{part}/facts-{index - index % SHARD:06d}.parquet'


def shard_path(root: str | Path, part: str, index: int) -> Path:
    """The local file of a shard; a shard of an HTTP store is fetched into the cache
    once. A missing file gives a path that does not exist."""
    root = str(root)
    name = shard_name(part, index)
    if not root.startswith(('http://', 'https://')):
        return Path(root) / name
    from . import data
    dest = data._cache_base(None) / 'facts' / hashlib.sha256(root.encode()).hexdigest()[:16] / name
    if not dest.exists():
        try:
            data._fetch(f'{root.rstrip("/")}/{name}', dest)
        except Exception:
            return dest
    return dest


@functools.lru_cache(maxsize=16)
def _shard(path: str) -> dict[int, str]:
    import pyarrow.parquet as pq
    t = pq.read_table(path, columns=['index', 'facts'])
    return dict(zip(t.column('index').to_pylist(), t.column('facts').to_pylist()))


def record(part: str, index: int, root: str | None = None) -> dict | None:
    """The stored record of one tile, or None when there is no store or no row."""
    root = root or facts_root()
    if not root:
        return None
    path = shard_path(root, part, index)
    if not path.exists():
        return None
    text = _shard(str(path)).get(index)
    return decode(text) if text is not None else None


def facts(part: str, index: int, caption: str | None = None, partition: str = 'all',
          root: str | None = None):
    """A tile's `TileFacts` from the store: no pixel is read."""
    from . import tasks as T
    rec = record(part, index, root)
    if rec is None:
        raise KeyError(f'no stored facts for {part} {index} under {root or facts_root()}')
    return T.TileFacts(cell=rec['cell'], sheet=rec['sheet'], caption=caption,
                       series=rec['series'], tile=None, partition=partition,
                       grids=rec['grids'])
