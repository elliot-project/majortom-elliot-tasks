"""The ELLIOT fact sheet: every measured fact about one tile, tagged by modality.

**The sheet is SEGMENTED BY MODALITY, because an encoder's input is not fixed.** A
model trained on ELLIOT will not always be fed the same cube: S1 has valid pixels for
~93% of samples, L8 can be dropped for speed, and an S2-only configuration is the
common case. A single prose blob asserting a thermal fact is then a label the input
cannot support -- the model is taught to hallucinate the missing sensor. So every
fact carries the modality that licenses it, and a fact whose modality is not in the
sample is dropped. `SEGMENTS` below is that contract for captions written one
paragraph per modality; `assemble()` is the consumer side of it.

**It carries SPECTRAL facts.** Place names, the admin chain, WorldCover, DEM and OSM
are all derivable from longitude, latitude and a gazetteer: a model could write a
caption from those without looking at the image. NDVI, water and built-up indices,
snow, burn ratio and L8's brightness temperature are facts that exist only in the
cube, and their 3x3 grid versions are the only facts in the sheet that say WHERE
inside the tile something is AND come from the pixels.

**Position is named, not numbered.** Features arrive with a 3x3 compass cell and,
for lines, the direction they run and the cells they cross. The 0-1 bbox stays in the
sheet for drawing and for grounding tasks, but the caption prompt forbids writing
numbers into prose: "a river running south-east across the lower left" is something a
model can learn to localise from, `[120, 340, 880, 910]` is not.

Every fact is committed to HERE, in code. A captioning model writes prose over
numbers it cannot choose; it is never handed raw tags.

Sources, all public: ELLIOT-Pretrain (S2, L8, DEM, WorldCover, the per-tile context
columns) and ELLIOT-X-EXT (cloud masks and fractions, ERA5 at each acquisition, S1,
OSM features and admin units, WorldCover histogram, DEM statistics).

Usage:
    import elliot_tasks as et
    et.configure(elliot='/path/to/elliot-pretrain', ext='/path/to/elliot-x-ext')
    sheet = et.fact_sheet(et.find('284D_496L'))
    print(sheet.text({'s2', 'l8', 's1', 'dem', 'lc'}))
"""
from __future__ import annotations

import collections
import json
import math
from dataclasses import dataclass, field

import numpy as np

from .data import TILE_PX, Tile, find

TILE_AREA_M2 = TILE_PX * TILE_PX * 100.0          # 111.5 km^2


def as_tile(tile_or_cell: Tile | str) -> Tile:
    return tile_or_cell if isinstance(tile_or_cell, Tile) else find(tile_or_cell)


# --------------------------------------------------------------------------------
# The modality contract.
# --------------------------------------------------------------------------------
#: One output paragraph per key. `requires` is the set of modalities that must be in
#: the encoder's input for that paragraph to be a licensed statement about it; a
#: paragraph whose requirement is unmet is DROPPED at assembly, never rewritten. `core`
#: requires nothing because geography, layout and named features are what a location
#: embedding and a map prior carry, and every encoder configuration has those.
SEGMENTS: dict[str, dict] = {
    'core':    {'requires': set(),
                'brief': 'Where this is, what kind of place it is, and how the scene is '
                         'laid out. True whatever sensors are present.'},
    'optical': {'requires': {'s2'},
                'brief': 'What the multispectral reflectance says: vegetation vigour and '
                         'where it is strongest, open water, bare ground, built-up '
                         'surfaces, snow, burn scars. Only facts from the spectral block.'},
    'thermal': {'requires': {'l8'},
                'brief': 'What Landsat-8 adds that Sentinel-2 cannot see: land-surface '
                         'temperature and where the scene is hottest or coolest. Omit if '
                         'no thermal facts are given.'},
    'radar':   {'requires': {'s1'},
                'brief': 'What C-band backscatter adds: smooth surfaces (open water, '
                         'tarmac, dry sand), volume scattering (forest), bright '
                         'double-bounce (built-up). Omit if no radar facts are given.'},
    'terrain': {'requires': {'dem'},
                'brief': 'Relief: how much of it there is and how it is arranged. Omit if '
                         'the tile is flat.'},
    'sky':     {'requires': {'s2'},
                'brief': 'Cloud, shadow and illumination, and what they hide. Omit when '
                         'the scene is essentially clear.'},
}


def assemble(segments: dict[str, str], available: set[str], order: list[str] | None = None) -> str:
    """Join the paragraphs this sample's modalities license, in a fixed order.

    The assembler is deliberately dumb. It does not rewrite, stitch or re-order within
    a paragraph, because any such edit would be an unmeasured claim about text a model
    wrote. A missing modality removes a paragraph; nothing else changes. That also
    means each paragraph must STAND ALONE, which is what the prompt asks for.
    """
    order = order or list(SEGMENTS)
    out = []
    for k in order:
        if k not in segments:
            continue
        txt = (segments[k] or '').strip()
        if not txt or txt.lower() in ('none', 'n/a', '-'):
            continue
        if SEGMENTS[k]['requires'] <= available:
            out.append(txt)
    return ' '.join(out)


# --------------------------------------------------------------------------------
# Named position. A 3x3 compass grid, because that is the resolution a caption can
# honestly assert and a model can plausibly learn.
# --------------------------------------------------------------------------------
CELL_NAMES = [['north-west', 'north', 'north-east'],
              ['west',       'centre', 'east'],
              ['south-west', 'south', 'south-east']]


def _cells_touched(x0: float, y0: float, x1: float, y1: float) -> list[str]:
    """Which of the nine compass cells a 0-1 bbox overlaps."""
    ci = lambda v: min(2, max(0, int(v * 3)))
    out = []
    for r in range(ci(y0), ci(y1) + 1):
        for c in range(ci(x0), ci(x1) + 1):
            out.append(CELL_NAMES[r][c])
    return out


def position_phrase(bbox01: tuple[float, float, float, float], is_line: bool,
                    geom=None) -> str:
    """Words, not numbers: where in the tile, and for a line which way it runs.

    A bbox is the wrong primitive for a line -- a diagonal river's bbox covers the
    tile while the river covers 1% -- so a line gets its bearing from its endpoints
    and its extent from the cells its vertices actually fall in, not from the corners
    of its envelope.
    """
    x0, y0, x1, y1 = bbox01
    w, h = x1 - x0, y1 - y0
    if w >= 0.85 and h >= 0.85:
        return 'across the whole tile'
    if is_line and geom is not None:
        pts = _line_points(geom)
        if len(pts) >= 2:
            cells, seen = [], set()
            for px, py in pts:
                n = CELL_NAMES[min(2, max(0, int(py * 3)))][min(2, max(0, int(px * 3)))]
                if n not in seen:
                    seen.add(n)
                    cells.append(n)
            dx, dy = pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1]
            bearing = _bearing(dx, dy)
            span = ' through '.join(cells) if len(cells) <= 4 else \
                   f'{cells[0]} to {cells[-1]}'
            return f'running {bearing}, {span}'
    cells = _cells_touched(x0, y0, x1, y1)
    if len(cells) == 1:
        return f'in the {cells[0]}'
    if len(cells) >= 7:
        return 'spread over most of the tile'
    return 'in the ' + ', '.join(cells[:-1]) + f' and {cells[-1]}'


def _bearing(dx: float, dy: float) -> str:
    """dy is positive DOWNWARD (image convention), so south is +y."""
    ang = math.degrees(math.atan2(dy, dx)) % 180.0
    if ang < 22.5 or ang >= 157.5:
        return 'east-west'
    if ang < 67.5:
        return 'north-west to south-east'
    if ang < 112.5:
        return 'north-south'
    return 'north-east to south-west'


def _line_points(geom) -> list[tuple[float, float]]:
    """Already-normalised 0-1 vertices of a line geometry, thinned to at most 8."""
    try:
        parts = list(geom.geoms) if hasattr(geom, 'geoms') else [geom]
    except Exception:
        return []
    pts = []
    for p in parts:
        if p.geom_type in ('LineString', 'LinearRing'):
            pts.extend(list(p.coords))
    if len(pts) > 8:
        step = len(pts) / 8.0
        pts = [pts[int(i * step)] for i in range(8)]
    return pts


# --------------------------------------------------------------------------------
# Spectral facts.
# --------------------------------------------------------------------------------
#: 0-based band indices into ELLIOT's S2 and L8 rasters (see each part's ml:contract).
S2 = dict(coastal=0, blue=1, green=2, red=3, re1=4, re2=5, re3=6, nir=7, nir08=8,
          wv=9, cirrus=10, swir16=11, swir22=12)
L8 = dict(coastal=0, blue=1, green=2, red=3, nir08=4, swir16=5, swir22=6, pan=7,
          cirrus=8, lwir11=9, lwir12=10)

#: Index thresholds. These are physical conventions, not dataset percentiles, and
#: that is a deliberate first cut: a percentile needs a reference distribution over
#: ELLIOT, and a threshold at least means the same thing in the Sahara and in
#: Amazonia. Where the two disagree -- an NDVI of 0.35 is sparse scrub globally but is
#: the greenest thing in a desert tile -- the caption should lean on the 3x3 grid
#: (which is internal to the tile) rather than the label.
NDVI_DENSE, NDVI_VEG, NDVI_BARE = 0.60, 0.30, 0.10
MNDWI_WATER, NDBI_BUILT, NDSI_SNOW, NBR_BURN = 0.20, 0.05, 0.40, -0.10


def _norm_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = a + b
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(np.abs(d) > 1e-6, (a - b) / d, np.nan)


def _grid3(arr: np.ndarray) -> np.ndarray:
    """Mean of a 2-D array over the 3x3 compass cells, ignoring NaN."""
    h, w = arr.shape
    rs, cs = h // 3, w // 3
    out = np.full((3, 3), np.nan)
    for r in range(3):
        for c in range(3):
            blk = arr[r * rs:(r + 1) * rs if r < 2 else h,
                      c * cs:(c + 1) * cs if c < 2 else w]
            if np.isfinite(blk).any():
                out[r, c] = np.nanmean(blk)
    return out


def _grid_phrase(g: np.ndarray, hi: str, lo: str, min_spread: float) -> str | None:
    """Turn a 3x3 map into one clause about WHERE, or nothing if it is uniform.

    The spread test matters more than the extremes: a tile whose NDVI runs 0.61 to
    0.66 is uniformly green, and saying it is "greenest in the south-east" invents a
    gradient out of noise. Only a spread above `min_spread` earns a direction.
    """
    if not np.isfinite(g).any():
        return None
    spread = float(np.nanmax(g) - np.nanmin(g))
    if spread < min_spread:
        return None
    hi_ix = np.unravel_index(np.nanargmax(g), g.shape)
    lo_ix = np.unravel_index(np.nanargmin(g), g.shape)
    return (f'{hi} in the {CELL_NAMES[hi_ix[0]][hi_ix[1]]}, '
            f'{lo} in the {CELL_NAMES[lo_ix[0]][lo_ix[1]]}')


def _pct(x: float) -> str:
    return f'{100 * x:.0f}%'


def _burn_is_plausible(sp: dict, wc: dict | None) -> bool:
    """A low NBR only means fire where there was something to burn.

    NBR separates healthy canopy from charred canopy, and it does that well. What it
    does not do is separate char from any other dark-in-NIR, bright-in-SWIR surface,
    which is most of a city: ungated, it reports a 13% "burn-scar signature" over
    central Curitiba, where WorldCover says 68% built-up. So the claim is gated on the
    tile being majority vegetated in the first place. This is a single-date proxy and
    stays a weak one -- a real burn statement wants dNBR against a pre-fire date,
    which the monthly and burst parts supply (`temporal.py`) and monotemporal cannot.
    """
    if not wc or sp.get('frac_low_nbr', 0.0) <= 0.05:
        return False
    hist = {int(k): v for k, v in wc.items()}
    tot = sum(hist.values()) or 1
    vegetated = sum(v for k, v in hist.items() if k in (10, 20, 30, 40, 95)) / tot
    return vegetated >= 0.5


def s2_spectral(tile: Tile, cloud_mask: np.ndarray | None = None,
                wc: dict | None = None, frame: int = 0) -> dict:
    """Cloud-masked index statistics from one Sentinel-2 acquisition.

    Cloud-masked because an index over cloud is an index of cloud: thick cloud has
    NDVI near zero and MNDWI near zero, so an unmasked 60%-cloudy forest tile reads
    as bare ground. Where the clear fraction falls below a fifth of the tile the
    statistics are returned but flagged, and the prompt tells the model to say the
    scene is obscured rather than to describe it.
    """
    raw = tile.frame('s2', frame)
    if raw is None or raw.shape[0] < 13:
        return {}
    cube = raw.astype(np.float32) * tile.scale('s2')
    valid = np.isfinite(cube).all(axis=0) & (cube[S2['red']] > 0)
    if cloud_mask is not None and cloud_mask.shape == valid.shape:
        valid &= cloud_mask
    if valid.sum() < 1000:
        return {'clear_px_fraction': float(valid.mean())}

    m = lambda k: np.where(valid, cube[S2[k]], np.nan)
    ndvi = _norm_diff(m('nir'), m('red'))
    mndwi = _norm_diff(m('green'), m('swir16'))
    ndbi = _norm_diff(m('swir16'), m('nir'))
    ndsi = _norm_diff(m('green'), m('swir16'))     # same algebra, different reading
    nbr = _norm_diff(m('nir'), m('swir22'))
    bright = np.nanmean(np.stack([m('blue'), m('green'), m('red')]), axis=0)

    # ORDER MATTERS, and getting it wrong is not subtle. Snow and open water BOTH
    # have a high green-minus-SWIR ratio -- snow absorbs SWIR almost completely --
    # so MNDWI alone cannot tell them apart: a December tile at Boucherville, Quebec,
    # a snow-covered suburb on a frozen river, reads as "94% open water". Snow is
    # resolved FIRST, on the NIR test that does separate them (snow stays bright in
    # NIR, water does not), and water is then whatever is left.
    snow = (ndsi > NDSI_SNOW) & (m('nir') > 0.11)
    water = (mndwi > MNDWI_WATER) & ~snow
    veg = ndvi > NDVI_VEG
    out = {
        'clear_px_fraction': float(valid.mean()),
        'ndvi_mean': float(np.nanmean(ndvi)),
        'ndvi_p10': float(np.nanpercentile(ndvi, 10)),
        'ndvi_p90': float(np.nanpercentile(ndvi, 90)),
        'frac_dense_veg': float(np.nanmean(ndvi > NDVI_DENSE)),
        'frac_veg': float(np.nanmean(veg)),
        'frac_bare': float(np.nanmean(ndvi < NDVI_BARE) - np.nanmean(water)),
        'frac_water': float(np.nanmean(water)),
        # NOT "built-up". NDBI is a SWIR-over-NIR ratio and bare soil has it too: it
        # calls a Mongolian gravel desert 100% built-up. What it honestly measures is
        # a non-vegetated surface that is bright in SWIR, which is bare ground AND
        # sealed ground. Separating the two needs WorldCover's classifier or S1's
        # double-bounce -- Curitiba is 46% radar-bright, the desert 0% -- and the
        # sheet carries both so the model can do it from the evidence rather than from
        # a mislabelled index.
        'frac_swir_bright_nonveg': float(np.nanmean((ndbi > NDBI_BUILT) & ~water & ~snow
                                                    & (ndvi < NDVI_VEG))),
        'frac_snow': float(np.nanmean(snow)),
        # NOT reported unconditionally: see `_burn_is_plausible`. A low NBR is a low
        # NIR-to-SWIR ratio, and asphalt and bare rock have that too.
        'frac_low_nbr': float(np.nanmean((nbr < NBR_BURN) & ~water & ~snow)),
        'brightness_mean': float(np.nanmean(bright)),
        'ndvi_grid': _grid3(ndvi),
        'water_grid': _grid3(water.astype(np.float32)),
        'snow_grid': _grid3(snow.astype(np.float32)),
        'swir_bright_grid': _grid3(((ndbi > NDBI_BUILT) & ~water & ~snow
                                    & (ndvi < NDVI_VEG)).astype(np.float32)),
    }
    out['frac_bare'] = max(0.0, out['frac_bare'])
    return out


def l8_thermal(tile: Tile, frame: int = 0) -> dict:
    """At-sensor brightness temperature from Landsat-8's TIRS band 10.

    The only fact in the whole sheet that no amount of optical or map data supplies,
    which is exactly why it is worth the read. ELLIOT stores brightness temperature
    in kelvin scaled by 1e4 (a band-10 value of 2,880,985 is 288.1 K); this is
    at-sensor brightness temperature, NOT emissivity-corrected LST, so it is reported
    as a relative pattern and a rounded number, never to a tenth of a degree.
    """
    raw = tile.frame('l8', frame)
    if raw is None or raw.shape[0] < 10:
        return {}
    bt = raw[L8['lwir11']].astype(np.float32) * tile.scale('l8') - 273.15
    bt = np.where((bt > -90) & (bt < 90), bt, np.nan)
    if not np.isfinite(bt).any():
        return {}
    return {
        'bt_mean_c': float(np.nanmean(bt)),
        'bt_p05_c': float(np.nanpercentile(bt, 5)),
        'bt_p95_c': float(np.nanpercentile(bt, 95)),
        'bt_grid': _grid3(bt),
    }


def s1_frame(tile: Tile, frame: int = 0) -> int | None:
    """Which S1 frame to describe: the one paired with S2 `frame` if it holds data,
    else the first that does. None when no frame has a VV+VH pass."""
    rows = tile.s1_frames
    ok = [k for k, r in enumerate(rows) if (r.get('s1:valid_fraction') or 0) > 0]
    if not ok:
        return None
    return frame if frame in ok else ok[0]


def s1_backscatter(tile: Tile, frame: int = 0) -> dict:
    """VV/VH in decibels, plus the three surface classes radar separates cleanly.

    Stored linear with nodata 0, so the conversion is 10*log10 and a zero is a hole,
    not a -inf. The thresholds are the conventional C-band ones: below -17 dB VV is a
    specular surface (calm water, tarmac, dry sand), and a VV-VH ratio under ~4 dB
    with high VH is volume scattering from canopy.
    """
    k = s1_frame(tile, frame)
    a = tile.frame('s1', k) if k is not None else None
    if a is None or a.shape[0] < 2:
        return {}
    vv, vh = a[0].astype(np.float32), a[1].astype(np.float32)
    ok = (vv > 0) & (vh > 0) & np.isfinite(vv) & np.isfinite(vh)
    if ok.sum() < 1000:
        return {}
    vv_db = np.where(ok, 10 * np.log10(np.where(ok, vv, 1)), np.nan)
    vh_db = np.where(ok, 10 * np.log10(np.where(ok, vh, 1)), np.nan)
    ratio = vv_db - vh_db
    return {
        'frame': k,
        'vv_db_mean': float(np.nanmean(vv_db)),
        'vh_db_mean': float(np.nanmean(vh_db)),
        'frac_specular': float(np.nanmean(vv_db < -17.0)),
        'frac_volume': float(np.nanmean((vh_db > -18.0) & (ratio < 4.0))),
        'frac_bright': float(np.nanmean(vv_db > -5.0)),
        'vv_grid': _grid3(vv_db),
    }


def cloud_record(tile: Tile, frame: int = 0) -> dict | None:
    """The OmniCloudMask fractions for one S2 acquisition (ELLIOT-X-EXT)."""
    rows = tile.s2_ext_frames
    if frame >= len(rows) or rows[frame].get('ml:clear') is None:
        return None
    r = rows[frame]
    return {'date': r.get('ml:date'), 'clear': r['ml:clear'],
            'thick_cloud': r['ml:thick_cloud'], 'thin_cloud': r['ml:thin_cloud'],
            'shadow': r['ml:shadow']}


def clear_mask(tile: Tile, frame: int = 0) -> np.ndarray | None:
    """Boolean 1056x1056: True where OmniCloudMask says clear (class 0).

    Loading the mask rather than trusting the scalar fraction is the point. A 30%
    cloudy tile still has 70% of a real scene in it, and masking recovers that;
    using the scalar alone leaves the choice between describing cloud as bare
    ground and discarding the tile.
    """
    a = tile.frame('cloud_mask_s2', frame)
    if a is None:
        return None
    return np.asarray(a) == 0


# --------------------------------------------------------------------------------
# OSM selection: which named features may reach the model (rules R1-R7).
# --------------------------------------------------------------------------------
KEEP_NODE = {
    'place': {'city', 'town', 'village', 'hamlet', 'suburb', 'quarter', 'borough',
              'municipality', 'island', 'islet', 'archipelago', 'locality',
              'isolated_dwelling'},
    'natural': {'peak', 'volcano', 'saddle', 'cape', 'spring', 'glacier', 'bay',
                'strait', 'fjord'},
    'man_made': {'lighthouse', 'works', 'water_tower', 'mast', 'tower', 'windmill',
                 'pier', 'dam'},
    'aeroway': {'aerodrome', 'airstrip'},
    'railway': {'station', 'halt'},
    'waterway': {'waterfall', 'dam'},
    'amenity': {'ferry_terminal'},
}
COARSE = {'region', 'continent', 'archipelago', 'peninsula', 'mountain_range', 'sea',
          'ocean', 'country', 'state', 'province', 'territory', 'subregion'}
GENERIC_BUILDING = {'yes', 'house', 'residential', 'hut', 'shed', 'garage', 'roof',
                    'service', 'detached', 'semidetached_house', 'apartments',
                    'static_caravan'}
NOMINAL_WIDTH_M = {'motorway': 30, 'trunk': 24, 'primary': 18, 'secondary': 14,
                   'tertiary': 10, 'residential': 8, 'service': 5, 'track': 4,
                   'path': 2, 'footway': 2, 'cycleway': 3, 'river': 40, 'stream': 6,
                   'canal': 20, 'ditch': 3, 'drain': 3, 'rail': 10}
MIN_AREA_M2, MIN_LINE_M = 4000.0, 200.0

#: A school is admitted when the tile has almost nothing else. It is invisible at
#: 10 m, but in a rural tile it is often the only named thing, and "no named
#: features" is a worse caption than "a school and a track". Kept behind a sparsity
#: gate so it never displaces real extent in a city.
SPARSE_NODE_EXTRA = {('amenity', 'school'), ('amenity', 'hospital'),
                     ('amenity', 'clinic'), ('amenity', 'place_of_worship'),
                     ('amenity', 'marketplace')}
SPARSE_THRESHOLD = 10

#: Dropped: a roadworks polygon is a real object and a poor descriptor of a 10 km
#: scene, and it dates the caption to the month.
DROP_SUBTYPE = {('landuse', 'construction')}


def salience(area_m2: float, length_m: float, subtype: str) -> float:
    a = area_m2 or 0.0
    if not a and length_m:
        a = length_m * NOMINAL_WIDTH_M.get(subtype, 6)
    return a / TILE_AREA_M2


def classify(f: dict, sparse: bool = False) -> str | None:
    """None if the feature reaches the model, else the rule that dropped it."""
    cat, sub, ot = f['category'], f['subtype'], f['osm_type']
    a, l = f['area'] or 0.0, f['length'] or 0.0
    if (cat, sub) in DROP_SUBTYPE:
        return 'R7 dated construction'
    if a >= 0.95 * TILE_AREA_M2 and (sub in COARSE or cat == 'place'):
        return 'R3 tile-filling locator'
    if ot == 'node' or (a == 0 and l == 0):
        if sub in KEEP_NODE.get(cat, ()):
            return None
        if sparse and (cat, sub) in SPARSE_NODE_EXTRA:
            return None
        return 'R2 role not on allow-list'
    if cat == 'building' and sub in GENERIC_BUILDING and a < 10000:
        return 'R4 generic building name'
    if a and a < MIN_AREA_M2:
        return 'R1 polygon below 4,000 m2'
    if l and l < MIN_LINE_M:
        return 'R1 line below 200 m'
    return None


def _clean(v):
    """A pandas missing value back to None."""
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def load_features(tile: Tile, named_only: bool = True, frame: int = 0) -> list[dict]:
    """OSM features for a tile, with 0-1 normalised geometry attached.

    Geometry is projected through the geotransform of the ELLIOT S2 RASTER: the
    raster a caption describes is the only authority for where a thing is in it.
    """
    from shapely import wkb
    from shapely.affinity import affine_transform

    df = tile.osm()
    if df is None or not tile.s2_frames:
        return []
    c, a_, _, f_, _, e = tile.geotransform(frame)
    # CRS metres -> 0-1 of the tile.
    A = [1.0 / (a_ * TILE_PX), 0.0, 0.0, 1.0 / (e * TILE_PX),
         -c / (a_ * TILE_PX), -f_ / (e * TILE_PX)]
    if named_only:
        df = df[df['named'].astype(bool)]
    out = []
    for r in df.itertuples(index=False):
        g = r.geometry
        G = wkb.loads(g) if g is not None else None
        Gn, bb = None, None
        if G is not None and not G.is_empty:
            Gn = affine_transform(G, A)
            x0, y0, x1, y1 = Gn.bounds
            bb = (x0, y0, x1, y1)
        sub = _clean(r.subtype)
        out.append(dict(name=_clean(r.name), category=_clean(r.category),
                        subtype=sub or '(none)', osm_type=r.osm_type,
                        area=_clean(r.area_in_tile_m2), length=_clean(r.length_in_tile_m),
                        bbox=bb, geom=Gn))
    return out


def merge_parts(kept: list[dict]) -> list[dict]:
    """R5: same name and subtype become one object, unless they are far apart."""
    from shapely.ops import unary_union
    groups: dict[tuple, list[dict]] = collections.defaultdict(list)
    for f in kept:
        groups[(f['name'], f['subtype'])].append(f)
    out = []
    for (nm, sub), parts in groups.items():
        if len(parts) == 1:
            f = dict(parts[0])
            f['n_parts'], f['merge'] = 1, 'single'
            out.append(f)
            continue
        area = sum(p['area'] or 0 for p in parts)
        length = sum(p['length'] or 0 for p in parts)
        geoms = [p['geom'] for p in parts if p['geom'] is not None]
        u = unary_union(geoms) if geoms else None
        bb, mode = parts[0]['bbox'], 'merged'
        if u is not None and not u.is_empty:
            x0, y0, x1, y1 = u.bounds
            hull = (x1 - x0) * (y1 - y0) * TILE_AREA_M2
            ext = area if area else max(length * 10.0, 1.0)
            if hull > 4 * ext:
                mode = 'grouped'
            bb = (x0, y0, x1, y1)
        out.append(dict(name=nm, category=parts[0]['category'], subtype=sub,
                        osm_type=parts[0]['osm_type'], area=area, length=length,
                        bbox=bb, n_parts=len(parts), merge=mode, geom=u))
    return out


def selected_objects(tile: Tile, top: int = 18) -> tuple[list[dict], list[dict]]:
    """(objects reaching the model, every named feature) for one tile."""
    feats = load_features(tile)
    sparse = sum(1 for f in feats if classify(f) is None) < SPARSE_THRESHOLD
    kept = [f for f in feats if classify(f, sparse) is None]
    objs = sorted(merge_parts(kept),
                  key=lambda m: -salience(m['area'], m['length'], m['subtype']))
    return objs[:top], feats


# --------------------------------------------------------------------------------
# Context blocks: locator, acquisition, land cover, terrain.
# --------------------------------------------------------------------------------
def place_line(tile: Tile) -> str | None:
    """Fine -> coarse, like a postal address, capped at level 8, with a span note.

    From the tile's OSM administrative units; where it has none, from ELLIOT's own
    country / state / district columns.
    """
    df = tile.admin()
    if df is not None and len(df):
        by = collections.defaultdict(list)
        for lvl, nm, a in zip(df['admin_level'], df['name'], df['area_in_tile_m2']):
            if _clean(nm) is None or _clean(lvl) is None:
                continue
            by[int(lvl)].append((nm, float(a or 0.0)))
        chain, spans = [], []
        for lvl in sorted(by):
            us = sorted(by[lvl], key=lambda x: -x[1])
            if lvl <= 8:
                chain.append(us[0][0])
            if lvl >= 8 and len(us) > 1:
                spans.append(f"{len(us)} units at level {lvl} "
                             f"({', '.join(n for n, _ in us[:3])}"
                             f"{', ...' if len(us) > 3 else ''})")
        seen, ordered = set(), []
        for c in reversed(chain):
            if c not in seen:
                seen.add(c)
                ordered.append(c)
        if ordered:
            line = 'LOCATION ' + ', '.join(ordered)
            return line + ('  [tile spans ' + '; '.join(spans) + ']' if spans else '')
    m = tile.meta
    if m.get('admin:country'):
        w = ', '.join(x for x in (m.get('admin:district'), m.get('admin:state'),
                                  m.get('admin:country'))
                      if x and x != 'Ocean/Sea/Lakes')
        return 'LOCATION ' + (w or 'open water, no administrative area')
    return None


WC = {10: 'tree cover', 20: 'shrubland', 30: 'grassland', 40: 'cropland',
      50: 'built-up', 60: 'bare or sparse vegetation', 70: 'snow and ice',
      80: 'permanent water', 90: 'herbaceous wetland', 95: 'mangroves',
      100: 'moss and lichen'}


def _worldcover_dem(tile: Tile) -> tuple[dict | None, dict | None]:
    """WorldCover histogram and DEM statistics over the tile (ELLIOT-X-EXT)."""
    wc, dem = tile.ext_meta.get('worldcover:hist_json'), \
        tile.ext_meta.get('terrain:dem_stats_json')
    return (json.loads(wc) if wc else None), (json.loads(dem) if dem else None)


def _acquisition(tile: Tile, frame: int = 0) -> dict:
    """Date and sun of one S2 acquisition, tile context, and ERA5 at overpass."""
    out: dict = {}
    if frame < len(tile.s2_frames):
        r = tile.s2_frames[frame]
        out['date'] = r.get('s2:date')
        out['spacecraft'] = r.get('s2:spacecraft')
        # Sun elevation is 90 minus the scene's mean solar zenith (S2 granule metadata).
        if r.get('s2:mean_solar_zenith') is not None:
            out['solar_elev'] = 90.0 - float(r['s2:mean_solar_zenith'])
    m = tile.meta
    out.update(human_mod=m.get('socio:human_modification'),
               climate_temp_k=m.get('climate:temperature'))
    if frame < len(tile.s2_ext_frames):
        e = tile.s2_ext_frames[frame]
        out.update(t2m=e.get('ml:era5_temperature_2m_c'),
                   dewpoint=e.get('ml:era5_dewpoint_2m_c'),
                   era5_cloud=e.get('ml:era5_cloud_cover_pct'),
                   precip=e.get('ml:era5_precipitation_mm'),
                   wind=e.get('ml:era5_wind_speed_10m_kmh'),
                   wind_dir=e.get('ml:era5_wind_from_direction_deg'))
    return out


def _human_mod_band(v: float) -> str:
    """The global human modification index, 0 (wild) to 1 (built)."""
    if v < 0.1:  return 'near-pristine, very little human modification'
    if v < 0.3:  return 'lightly modified by human activity'
    if v < 0.6:  return 'substantially modified by human activity'
    return 'heavily modified by human activity'


def _rh_from_dewpoint(t_c: float, td_c: float) -> float:
    """Relative humidity by the Magnus formula, from temperature and dewpoint."""
    a, b = 17.625, 243.04
    return 100.0 * math.exp(a * td_c / (b + td_c) - a * t_c / (b + t_c))


# --------------------------------------------------------------------------------
# The sheet.
# --------------------------------------------------------------------------------
@dataclass
class FactSheet:
    cell: str
    blocks: list[tuple[str, str]] = field(default_factory=list)   # (modality, line)
    objects: list[dict] = field(default_factory=list)
    available: set[str] = field(default_factory=set)
    raw: dict = field(default_factory=dict)

    def text(self, available: set[str] | list[str] | None = None) -> str:
        av = set(available) if available is not None else self.available
        keep = [ln for mod, ln in self.blocks if mod == '' or mod in av]
        return '\n'.join(keep)

    def segment_menu(self, available: set[str] | list[str] | None = None) -> str:
        av = set(available) if available is not None else self.available
        rows = []
        for k, spec in SEGMENTS.items():
            if spec['requires'] <= av:
                rows.append(f'  "{k}": {spec["brief"]}')
        return '\n'.join(rows)


def fact_sheet(tile: Tile | str, available: set[str] | None = None, top: int = 18,
               frame: int = 0) -> FactSheet:
    """Every fact, each tagged with the modality that licenses it.

    `available` defaults to what this tile actually HAS. Pass a smaller set to
    simulate an encoder configuration -- dropping S1 and L8 shrinks the sheet, and
    a caption written from it, together.

    For a monthly or burst tile the single-date facts describe acquisition `frame`
    (default the first); the series facts are in `temporal.py`.
    """
    tile = as_tile(tile)
    have = {leaf for leaf in ('s2', 'l8', 'dem', 'lc', 's1') if tile.has(leaf)}
    av = have if available is None else (set(available) & have)

    fs = FactSheet(cell=tile.cell, available=av)
    B = fs.blocks.append
    em = tile.ext_meta

    # --- always-true context ---------------------------------------------------
    ll = tile.lonlat
    if ll is not None:
        lon, lat = ll
        fs.raw.update(lat=lat, lon=lon, n_named=em.get('osm:n_named'),
                      n_unnamed=em.get('osm:n_unnamed'), n_nodes=em.get('osm:n_nodes'))
        crs = tile.crs(frame) if tile.s2_frames else ''
        B(('', f'GRID CELL {tile.cell}  centre {lat:.4f}, {lon:.4f}  {crs}  '
              f'1056x1056 px at 10 m (10.56 x 10.56 km)'))
    pl = place_line(tile)
    if pl:
        B(('', pl))

    acq = _acquisition(tile, frame)
    fs.raw['acq'] = acq
    if acq.get('date'):
        line = f'ACQUIRED {acq["date"]}'
        if acq.get('solar_elev') is not None:
            line += f'  solar elevation {acq["solar_elev"]:.0f} deg'
        B(('', line))
        if acq.get('t2m') is not None:
            w = f'WEATHER AT OVERPASS (ERA5) {acq["t2m"]:.0f} C'
            if acq.get('dewpoint') is not None:
                w += f', {_rh_from_dewpoint(acq["t2m"], acq["dewpoint"]):.0f}% RH'
            if acq.get('precip') is not None:
                w += f', {acq["precip"]:.1f} mm precip'
            if acq.get('wind') is not None:
                w += f', wind {acq["wind"]:.0f} km/h'
            if acq.get('era5_cloud') is not None:
                w += f', ERA5 cloud cover {acq["era5_cloud"]:.0f}%'
            B(('', w))
    # Context columns ELLIOT carries for every tile, banded rather than quoted.
    # `climate:precipitation` is left out: its unit is not documented with the data,
    # and a wrongly-labelled rainfall figure is worse than no rainfall figure.
    #
    # `socio:population` is DELIBERATELY NOT EMITTED. Read as people/km2 it puts
    # Curitiba -- 68% built-up, 9,577 named OSM features -- at 74/km2, i.e. "rural to
    # peri-urban", while a Mongolian desert correctly reads 0. The column is either in
    # other units or averaged over a much coarser cell than the 10 km tile.
    ctx_bits = []
    if acq.get('human_mod') is not None:
        ctx_bits.append(_human_mod_band(acq['human_mod']))
    if acq.get('climate_temp_k'):
        ctx_bits.append(f'long-term mean annual temperature about '
                        f'{acq["climate_temp_k"] - 273.15:.0f} C')
    if ctx_bits:
        B(('', 'SETTLEMENT AND CLIMATE CONTEXT ' + '; '.join(ctx_bits)))

    # --- land cover: a WorldCover fact, but readable off S2 too -----------------
    wc, dem = _worldcover_dem(tile)
    fs.raw['worldcover'], fs.raw['dem'] = wc, dem
    if wc:
        hist = {int(k): v for k, v in wc.items()}
        tot = sum(hist.values()) or 1
        parts = [f'{WC.get(k, f"class {k}")} {100 * v / tot:.0f}%'
                 for k, v in sorted(hist.items(), key=lambda kv: -kv[1])
                 if 100 * v / tot >= 1.0]
        if parts:
            B(('', 'LAND COVER (ESA WorldCover) ' + '; '.join(parts)))
    if dem:
        B(('dem', f'ELEVATION (Copernicus DEM) {dem["min"]:.0f} to {dem["max"]:.0f} m '
                  f'(median {dem["p50"]:.0f} m, relief {dem["max"] - dem["min"]:.0f} m)'))

    # --- cloud ------------------------------------------------------------------
    cl = cloud_record(tile, frame)
    fs.raw['cloud'] = cl
    cmask = clear_mask(tile, frame) if 's2' in av else None
    if cl:
        B(('s2', f'CLOUD (OmniCloudMask, Sentinel-2) clear {_pct(cl["clear"])}, '
                 f'thick cloud {_pct(cl["thick_cloud"])}, '
                 f'thin cloud {_pct(cl["thin_cloud"])}, shadow {_pct(cl["shadow"])}'))

    # --- spectral ---------------------------------------------------------------
    if 's2' in av:
        sp = s2_spectral(tile, cmask, wc, frame)
        fs.raw['s2_spectral'] = sp
        if sp.get('ndvi_mean') is not None:
            lines = [
                f'vegetation index NDVI mean {sp["ndvi_mean"]:+.2f} '
                f'(10th-90th pct {sp["ndvi_p10"]:+.2f} to {sp["ndvi_p90"]:+.2f})',
                f'surface fractions: vegetated {_pct(sp["frac_veg"])} '
                f'(densely {_pct(sp["frac_dense_veg"])}), open water '
                f'{_pct(sp["frac_water"])}, unvegetated and SWIR-bright '
                f'(bare ground or sealed surfaces) {_pct(sp["frac_swir_bright_nonveg"])}'
                + (f', snow or ice {_pct(sp["frac_snow"])}'
                   if sp['frac_snow'] > 0.02 else '')
                + (f', burn-scar signature {_pct(sp["frac_low_nbr"])}'
                   if _burn_is_plausible(sp, wc) else ''),
            ]
            # A gradient of something that is not there is noise dressed as a fact:
            # an Antarctic tile, NDVI -0.22 throughout, would be "greenest in the
            # north".
            g = _grid_phrase(sp['ndvi_grid'], 'greenest', 'least green', 0.10)
            if g and sp['frac_veg'] > 0.10:
                lines.append(f'vegetation gradient: {g}')
            g = _grid_phrase(sp['water_grid'], 'most water', 'least', 0.15)
            if g and sp['frac_water'] > 0.03:
                lines.append(f'water distribution: {g}')
            g = _grid_phrase(sp['swir_bright_grid'], 'most concentrated', 'least', 0.15)
            if g and sp['frac_swir_bright_nonveg'] > 0.03:
                lines.append(f'unvegetated surfaces: {g}')
            if sp['frac_snow'] > 0.02:
                g = _grid_phrase(sp['snow_grid'], 'deepest cover', 'least', 0.15)
                if g:
                    lines.append(f'snow and ice: {g}')
            B(('s2', 'SPECTRAL (Sentinel-2, cloud-masked) ' + '; '.join(lines)))

    if 'l8' in av:
        th = l8_thermal(tile, frame)
        fs.raw['l8_thermal'] = th
        if th:
            line = (f'THERMAL (Landsat-8 TIRS, at-sensor brightness temperature; '
                    f'a separate overpass from the Sentinel-2 date above) '
                    f'{th["bt_mean_c"]:.0f} C mean, '
                    f'{th["bt_p05_c"]:.0f} to {th["bt_p95_c"]:.0f} C over the tile')
            g = _grid_phrase(th['bt_grid'], 'warmest', 'coolest', 1.5)
            if g:
                line += f'; {g}'
            B(('l8', line))

    if 's1' in av:
        sar = s1_backscatter(tile, frame)
        fs.raw['s1'] = sar
        if sar:
            line = (f'RADAR (Sentinel-1 RTC, C-band) VV {sar["vv_db_mean"]:.1f} dB, '
                    f'VH {sar["vh_db_mean"]:.1f} dB mean; '
                    f'specular/smooth surfaces {_pct(sar["frac_specular"])}, '
                    f'volume scattering (canopy-like) {_pct(sar["frac_volume"])}, '
                    f'bright double-bounce (built-up-like) {_pct(sar["frac_bright"])}')
            g = _grid_phrase(sar['vv_grid'], 'brightest', 'darkest', 2.0)
            if g:
                line += f'; {g}'
            B(('s1', line))

    # --- OSM --------------------------------------------------------------------
    objs, _ = selected_objects(tile, top)
    fs.objects = objs
    if fs.raw.get('n_named') is not None:
        B(('', f'OSM FEATURE COUNTS {fs.raw["n_named"]} named, '
              f'{fs.raw["n_unnamed"]} unnamed above the 4,000 m2 area / 200 m length '
              f'gate, {fs.raw["n_nodes"]} named POI nodes'))
    if objs:
        ns = []
        for m in objs:
            a, l = m['area'] or 0, m['length'] or 0
            size = (f'{a / 1e4:.0f} ha' if a >= 1e5 else f'{a / 1e4:.1f} ha') \
                if a >= 4000 else (f'{l / 1000:.1f} km' if l >= 200 else '')
            tag = m['subtype'] if m['subtype'] != '(none)' else m['category']
            is_line = bool(l) and not a
            pos = position_phrase(m['bbox'], is_line, m['geom']) if m['bbox'] else ''
            bits = [tag] + ([size] if size else []) + ([pos] if pos else [])
            if m.get('n_parts', 1) > 1:
                bits.append(f'{m["n_parts"]} parts')
            ns.append(f'{m["name"]} ({", ".join(bits)})')
        B(('', 'NAMED FEATURES IN TILE, largest first — ' + '; '.join(ns)))
    cats = collections.Counter()
    df = tile.osm()
    if df is not None:
        # Counted in (category, subtype) order, missing subtype first, so that ties
        # in `most_common` come out in a stable, alphabetical order.
        pairs = sorted(((_clean(c), _clean(s)) for c, s in zip(df['category'], df['subtype'])),
                       key=lambda p: (p[0] or '', p[1] is not None, p[1] or ''))
        for cat, sub in pairs:
            cats[f'{cat}/{sub}'] += 1
    if cats:
        B(('', 'FEATURE MIX ' + '; '.join(f'{k} x{v}' for k, v in cats.most_common(12))))
    return fs


# --------------------------------------------------------------------------------
# A caption prompt over the sheet: one paragraph per segment, as JSON.
# --------------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You write captions for satellite image tiles, for training an Earth-observation \
vision-language model.

You are given only MEASURED METADATA -- never the image. Every number below was \
measured from the pixels or read from a map; your job is prose, not inference.

You return a JSON object. Each key is one PARAGRAPH, and each paragraph is licensed by \
a different sensor. At training time the model sees only the sensors a given sample \
carries, and only the paragraphs those sensors license are kept -- so every paragraph \
must STAND ALONE and must not refer to any other paragraph ("as noted above", "this \
same area", "in addition" are all forbidden). Do not repeat a fact across paragraphs.

Rules for all paragraphs:
- Plain declarative prose. No preamble, no bullets, no markdown, no headings.
- Use ONLY the facts given. Never invent a settlement, road, building, season, crop or \
activity that is not listed.
- Describe what a 10 m sensor would actually resolve over a 10.56 x 10.56 km area. A \
sub-pixel point of interest may be named as a landmark, never as a visible object.
- Say WHERE things are, using the compass words in the facts (north-west, centre, \
south-east, ...). Never write a coordinate, a pixel index, a bounding box or a CRS.
- Do not restate percentages or index values verbatim. Turn them into proportions in \
words: "most of", "about a third", "a narrow strip".
- If the clear fraction is low, say the scene is largely obscured rather than \
describing surfaces the cloud is covering.
- A paragraph with nothing to say is an empty string. Never pad."""

USER_SUFFIX = """\
Return a JSON object with exactly these keys and no others:
{keys}

{menu}

Length: "core" is 2-4 sentences; every other paragraph is 1-2 sentences. Output the \
JSON object and nothing else."""


def build_user_prompt(fs: FactSheet, available: set[str] | None = None) -> str:
    av = set(available) if available is not None else fs.available
    keys = [k for k, s in SEGMENTS.items() if s['requires'] <= av]
    return (fs.text(av) + '\n\n' +
            USER_SUFFIX.format(keys=json.dumps(keys), menu=fs.segment_menu(av)))
