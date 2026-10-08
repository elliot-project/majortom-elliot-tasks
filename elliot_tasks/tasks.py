"""Deterministic, on-the-fly task construction over an ELLIOT tile.

NOTHING IS STORED. A task is built when the loader asks for it, from the measured
fact sheet and from clipped OSM geometry, seeded on `(cell, family, epoch)`. The
curriculum can change without re-deriving 279,166 tiles, the model cannot memorise
one frozen string per tile because the wording is resampled, and boxes, grids and
captions come out of one code path instead of three artifacts that drift.

Captions are built the same way, by the rule-based captioner in `caption.py`, with a
fresh wording per epoch and only the paragraphs the sample's input modalities license.
A caller with its own captions (from an LLM, say) can pass them instead, and the
`caption` family then reads that stored string.

The design carries over lessons from an earlier task suite ("prior suite" below):
the unified integer-grid schema, the paraphrase banks, the train/eval split BY
TEMPLATE INDEX, `chain_order` and `pack_conversation`, and evidence-before-answer
targets. Its seven contract fixes are applied here from the start:

  1. one grid family, one schema, one parser -- no parallel categorical grid
  2. every grid prompt STATES its cell-coverage threshold (0.5 default, 0.2 cloud)
  3. population-like grids sum, never average
  4. no `class_fraction` family -- mostly-empty masks, poorly duplicated
  5. no yes/no `presence` family -- `which_present` keeps location and scored
     0.5402 -> 0.9425 where yes/no did not
  6. `change` keeps genuine no-change cases but caps them
  7. prompts and scorers move together (`scoring.py`)

NOT HERE: referring expressions, which need a relation derivation over OSM geometry.
The registry is the contract; filling it is mechanical.
"""
from __future__ import annotations

import json
import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import caption as capt
from . import factsheet as fsheet
from . import temporal as tmp
from .data import Tile
from .shapes import SHAPES, Feasibility, Shape, ShapeSpec, derived_spec

# --------------------------------------------------------------------------------
# Serialisation. Coordinates are integers on [0, 1000] IN TEXT ONLY -- the data
# store pixels and metres. The quantiser is the boundary between the two and is the
# only place the 1000 appears.
# --------------------------------------------------------------------------------
GRID_UNKNOWN = None
GRID_MIN_SIDE, GRID_MAX_SIDE, GRID_MAX_CELLS = 3, 12, 100
DEFAULT_CELL_THRESHOLD = 0.5
CLOUD_CELL_THRESHOLD = 0.2      # thin cloud vanishes under majority voting


def q1000(frac: float) -> int:
    """A 0-1 tile fraction to the integer grid the text uses."""
    return int(round(min(max(frac, 0.0), 1.0) * 1000))


def _json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(',', ':'))


def point_target(x: float, y: float, label: str) -> str:
    return _json({'point_2d': [q1000(x), q1000(y)], 'label': label})


def bbox_target(bbox01: Sequence[float], label: str) -> str:
    x0, y0, x1, y1 = bbox01
    return _json({'bbox_2d': [q1000(x0), q1000(y0), q1000(x1), q1000(y1)],
                  'label': label})


def points_multi_target(points01: Sequence[Sequence[float]], label: str,
                        row_band: int = 60) -> str:
    """One point per instance, in banded raster order.

    Scoring is order-invariant via greedy matching, but the TARGET is emitted in a
    deterministic order or the sequence is not learnable at all. Rows are banded
    before sorting by x, because a strict y sort makes two side-by-side objects
    swap places over a one-pixel difference in centroid.
    """
    qs = [(q1000(x), q1000(y)) for x, y in points01]
    qs.sort(key=lambda p: (p[1] // row_band, p[0]))
    return _json({'points_2d': [[x, y] for x, y in qs], 'label': label})


def grid_target(variable: str, grid: Sequence[Sequence[int | None]]) -> str:
    rows, cols = len(grid), len(grid[0])
    return _json({'variable': variable, 'shape': [rows, cols],
                  'grid': [list(r) for r in grid]})


def scalar_target(value: Any) -> str:
    if isinstance(value, bool):
        return 'yes' if value else 'no'
    if isinstance(value, float):
        return json.dumps(round(value, 2))
    if isinstance(value, (list, tuple)):
        return _json(list(value))
    return str(value)


def record_target(**values: Any) -> str:
    return json.dumps(values, ensure_ascii=False, separators=(',', ':'),
                      sort_keys=True)


def derived_target(evidence_key: str, evidence: Any,
                   answer_key: str, answer: Any) -> str:
    """Evidence first, answer last. Key order IS the reasoning order and is kept."""
    return _json({evidence_key: evidence, answer_key: answer})


# --------------------------------------------------------------------------------
# Wording. The split is BY TEMPLATE INDEX, not by example, so an evaluation phrasing
# cannot leak into training even when the corpus is regenerated from the same cells.
# --------------------------------------------------------------------------------
PROMPTS: dict[str, tuple[str, ...]] = {
    'caption': (
        'Describe this scene.', 'Write a short description of this tile.',
        'What does this satellite image show?', 'Summarise what is visible here.',
        'Give a brief account of this area.', 'Describe the landscape in this tile.',
        'In a few sentences, what is this place like?',
        'Characterise this scene for a reader who cannot see it.',
        'Report what this observation shows.', 'Describe the area covered here.',
    ),
    'dominant_cover': (
        'Which land-cover class covers the largest share of this tile?',
        'What is the dominant land cover here?',
        'Name the most extensive land-cover class in this scene.',
        'Which single class occupies most of this tile?',
        'State the prevailing land cover.',
        'What covers most of the ground in this image?',
        'Identify the majority land-cover class.',
        'Which class dominates this scene?',
        'Give the most common land cover in this tile.',
        'What is the principal surface type here?',
    ),
    'which_present': (
        'Exactly one of these features lies inside this tile. Which one?',
        'Which of these named features is in this scene?',
        'One of the following is present here. Which?',
        'Pick the feature that actually appears in this tile.',
        'Which option names something inside this footprint?',
        'Select the feature contained in this image.',
        'Only one of these is here. Which is it?',
        'Identify the option that is present in this scene.',
        'Which of these can be found in this tile?',
        'Choose the feature this tile contains.',
    ),
    'locate_point': (
        'Point to {label}.', 'Locate {label} with a single point.',
        'Mark the centre of {label}.', 'Where is {label}? Give one point.',
        'Place a point on {label}.', 'Indicate where {label} is.',
        'Pinpoint {label} in this tile.', 'Return one coordinate on {label}.',
        'Show the location of {label}.', 'Give a point that lies on {label}.',
    ),
    'locate_bbox': (
        'Give the bounding box of {label}.', 'Box {label}.',
        'Where is {label}? Return a bounding box.',
        'Draw a bounding box around {label}.',
        'Enclose {label} in a rectangle.', 'Return the extent of {label} as a box.',
        'Bound {label} with a rectangle.', 'Locate {label} with a box.',
        'Give the rectangular extent of {label}.', 'Frame {label}.',
    ),
    'features_all': (
        'Point to each named feature in this tile.',
        'Give one point per named feature listed here.',
        'Mark every named feature in this scene.',
        'Locate all the named features.',
        'Return a point for each named object in this tile.',
        'Where is each named feature? One point each.',
        'Identify the position of every named feature.',
        'Place one point on each named feature.',
        'Give the locations of all named features here.',
        'Mark the named features in this image.',
    ),
    'ndvi_grid': (
        'Report vegetation greenness across the tile as a grid.',
        'Give NDVI on a grid over this scene.',
        'Map vegetation vigour cell by cell.',
        'Produce a grid of NDVI values for this tile.',
        'How green is each part of this scene? Answer as a grid.',
        'Return per-cell vegetation index values.',
        'Describe greenness spatially, as a grid.',
        'Give the NDVI distribution as a lattice.',
        'Report vegetation index per grid cell.',
        'Grid this tile and report NDVI in each cell.',
    ),
    'cover_grid': (
        'Report land cover across the tile as a grid.',
        'Give the land-cover class of each grid cell.',
        'Map land cover cell by cell.',
        'Produce a land-cover grid for this scene.',
        'Which class dominates each cell? Answer as a grid.',
        'Return the per-cell land-cover classes.',
        'Describe land cover spatially, as a grid.',
        'Grid this tile and name the cover class in each cell.',
        'Give the land-cover lattice for this image.',
        'Report the dominant class per grid cell.',
    ),
    'vegetated_fraction': (
        'What share of this tile is vegetated?',
        'How much of this scene carries vegetation?',
        'Give the vegetated fraction of this tile.',
        'What percentage of the ground here is vegetated?',
        'Estimate the vegetation cover of this scene.',
        'How much of this image is green?',
        'Report the proportion of vegetated surface.',
        'What fraction of the tile shows vegetation?',
        'Give the percentage under vegetation.',
        'How extensive is the vegetation here?',
    ),
    'acquisition': (
        'When was this observed, and how high was the sun?',
        'Report the acquisition date and solar elevation.',
        'Give the observation metadata for this scene.',
        'What are the acquisition date and sun angle?',
        'State when this image was taken and the solar elevation.',
        'Report date and illumination geometry.',
        'Give the capture date and the sun elevation.',
        'When was this acquired? Include the solar elevation.',
        'Summarise the acquisition conditions.',
        'What date and sun angle produced this observation?',
    ),
    'peak_greenness': (
        'In which month is this area greenest, typically?',
        'When in the year does vegetation here peak?',
        'Name the month of peak greenness.',
        'Which month shows the most vigorous vegetation?',
        'At what point in the seasonal cycle is this area greenest?',
        'Give the month when vegetation is at its maximum.',
        'Which of these months is the greenest here?',
        'When does this landscape green up most?',
        'Identify the peak month for vegetation.',
        'Which month carries the highest vegetation index?',
    ),
    'green_up_count': (
        'How many separate green-up periods does this area show in a year?',
        'How many growing seasons are visible in this seasonal cycle?',
        'Count the distinct green-up periods here.',
        'Does this area green up once a year, or more than once?',
        'How many vegetation peaks does the seasonal cycle contain?',
        'Give the number of separate growing periods.',
        'How many times a year does vegetation rise here?',
        'Count the growing seasons in this cycle.',
        'How many green-up events does this area show?',
        'State the number of distinct growing periods.',
    ),
    'clearest_frame': (
        'Which acquisition in this series is the least cloudy?',
        'Which frame gives the clearest view of the ground?',
        'Identify the clearest acquisition.',
        'Which of these dates is least obscured by cloud?',
        'Pick the frame with the best visibility.',
        'Which acquisition would you use to see the surface?',
        'Name the clearest date in this series.',
        'Which frame has the least cloud?',
        'Select the acquisition with the clearest sky.',
        'Which of these observations is the most usable?',
    ),
    'ndvi_series': (
        'Give the vegetation index for each acquisition in order.',
        'Report NDVI across the series.',
        'What is the greenness of each frame?',
        'List the vegetation index per acquisition.',
        'Trace vegetation through this series.',
        'Give the per-frame vegetation values.',
        'Report the greenness trajectory.',
        'What does the vegetation index do across these frames?',
        'Give NDVI for every acquisition, in order.',
        'Return the vegetation index series.',
    ),
    'disturbance': (
        'Did an abrupt surface change occur in this series, and when?',
        'Identify any sudden change in this window and the date it happened.',
        'When, if ever, does the surface change abruptly here?',
        'Report any fire, flood or clearance in this series and its date.',
        'Find the date of the largest abrupt surface change.',
        'Which acquisition follows a sudden change on the ground?',
        'When does this scene change sharply?',
        'Identify the frame after the biggest disturbance.',
        'Report the date of abrupt change in this window.',
        'Which date marks a sudden surface disturbance?',
    ),
    'relief': (
        'Describe the relief of this tile.',
        'How much elevation change is there here?',
        'Report the terrain relief.',
        'Give the elevation range of this scene.',
        'Is this tile flat or hilly? Quantify it.',
        'Summarise the topography.',
        'What is the vertical extent of this landscape?',
        'Report how much the ground rises and falls here.',
        'Characterise the terrain of this tile.',
        'Give the relief in metres.',
    ),
}


def template_indices(family: str, partition: str = 'all') -> tuple[int, ...]:
    """The stable 80/20 train/eval split, by template INDEX.

    By index and not by example: the corpus is regenerated on the fly, so a split
    over sampled records would put the same wording on both sides the next time the
    seed changed.
    """
    size = len(PROMPTS[family])
    n_eval = max(1, math.ceil(size * 0.2))
    if partition == 'train':
        return tuple(range(size - n_eval))
    if partition == 'eval':
        return tuple(range(size - n_eval, size))
    if partition == 'all':
        return tuple(range(size))
    raise ValueError('partition must be train, eval or all')


FORMAT_INSTRUCTIONS: dict[Shape, str] = {
    Shape.PROSE: 'Answer in two to four sentences of plain prose.',
    Shape.WORD: 'Answer with exactly one name copied from the list, and nothing else.',
    Shape.CHOICE: 'Answer with one option copied exactly, and nothing else.',
    Shape.NUMBER: 'Answer with a single number and no unit.',
    Shape.NUMBER_ARRAY: 'Answer only a JSON array of numbers.',
    Shape.ORDER: 'Answer only a JSON array giving the order.',
    Shape.POINT: ('Answer only JSON: {"point_2d":[x,y],"label":"..."}, with x and y '
                  'integers from 0 to 1000 measured across and down the image.'),
    Shape.BBOX: ('Answer only JSON: {"bbox_2d":[xmin,ymin,xmax,ymax],"label":"..."}, '
                 'with integers from 0 to 1000.'),
    Shape.POINTS_MULTI: ('Answer only JSON: {"points_2d":[[x,y],...],"label":"..."}, '
                         'with integers from 0 to 1000, one point per feature.'),
    Shape.BOXES_MULTI: ('Answer only JSON: '
                        '{"boxes_2d":[[xmin,ymin,xmax,ymax],...],"label":"..."}.'),
    Shape.VERTICES: 'Answer only JSON: {"polygon_2d":[[x,y],...],"label":"..."}.',
    Shape.GRID: ('Answer only JSON: {"variable":"...","shape":[rows,cols],'
                 '"grid":[[value,...],...]}, one inner array per row, row-major from '
                 'the north-west corner. Use null for a cell with no value.'),
    Shape.RECORD: 'Answer only a JSON object with the named keys.',
}

LABEL_SYNONYMS: dict[str, tuple[str, ...]] = {
    'tree cover': ('forest', 'woodland', 'tree canopy'),
    'shrubland': ('scrub', 'bushland', 'scrubland'),
    'grassland': ('grass', 'pasture', 'meadow'),
    'cropland': ('farmland', 'arable land', 'cultivated land', 'fields'),
    'built-up': ('built-up land', 'urban fabric', 'developed land'),
    'bare or sparse vegetation': ('bare ground', 'sparsely vegetated ground'),
    'snow and ice': ('snow cover', 'ice'),
    'permanent water': ('open water', 'water'),
    'herbaceous wetland': ('wetland', 'marsh'),
    'mangroves': ('mangrove forest', 'mangrove swamp'),
    'moss and lichen': ('moss', 'lichen cover'),
}
SYNONYM_RATE = 0.25


def paraphrase(label: str, rng: random.Random, *, rate: float = SYNONYM_RATE) -> str:
    alts = LABEL_SYNONYMS.get(label)
    if alts and rng.random() < rate:
        return alts[rng.randrange(len(alts))]
    return label


def option_list(truth: str, rng: random.Random, pool: Sequence[str], *,
                count: int = 6) -> tuple[str, list[str]]:
    """A shuffled option list containing the truth, and the truth AS OFFERED.

    The answer string is returned because `paraphrase` may have renamed it. Grading
    against the original name when the model was shown a synonym is grading a copy
    of the wrong vocabulary.
    """
    answer = paraphrase(truth, rng)
    banned = {truth, answer, *LABEL_SYNONYMS.get(truth, ())}
    candidates = [c for c in pool if c not in banned]
    rng.shuffle(candidates)
    options = [answer] + candidates[:max(1, count - 1)]
    rng.shuffle(options)
    return answer, options


# --------------------------------------------------------------------------------
# The registry.
# --------------------------------------------------------------------------------
@dataclass(frozen=True)
class Family:
    name: str
    shape: Shape
    #: ELLIOT leaves that must be in the encoder's input for this task to be a
    #: licensed question about it. Same vocabulary as `factsheet.SEGMENTS`.
    requires: frozenset[str]
    build: Callable[[TileFacts, random.Random], Example | None]
    #: Where it sits in a conversation: dense spatial output first, naming last, so
    #: a later turn can refer back to something already established.
    chain_stage: int = 1
    #: Required when `shape` is DERIVED, forbidden otherwise: (evidence, answer).
    #: DERIVED is a wrapper, not an entry in SHAPES, so its spec is COMPUTED from
    #: the two shapes it wraps -- which is the only way the promotion rule in
    #: `derived_spec` can apply.
    derived_from: tuple[Shape, Shape] | None = None

    def __post_init__(self):
        if (self.shape is Shape.DERIVED) != (self.derived_from is not None):
            raise ValueError(f'{self.name}: DERIVED needs derived_from, and only '
                             f'DERIVED may have it')

    @property
    def spec(self) -> ShapeSpec:
        if self.shape is Shape.DERIVED:
            return derived_spec(*self.derived_from)
        return SHAPES[self.shape]

    @property
    def feasibility(self) -> Feasibility:
        return self.spec.feasibility


@dataclass
class Example:
    family: str
    question: str
    target: str
    shape: Shape
    meta: dict = field(default_factory=dict)


FAMILIES: dict[str, Family] = {}


def register(name: str, shape: Shape, requires: Sequence[str], *,
             chain_stage: int = 1,
             derived_from: tuple[Shape, Shape] | None = None):
    def deco(fn):
        FAMILIES[name] = Family(name, shape, frozenset(requires), fn, chain_stage,
                                derived_from)
        return fn
    return deco


# --------------------------------------------------------------------------------
# Tile facts: one read, reused by every family for that tile.
# --------------------------------------------------------------------------------
@dataclass
class TileFacts:
    cell: str
    sheet: fsheet.FactSheet
    caption: str | None = None       # a stored caption, when the caller has one
    #: The encoder's input modalities for this draw, set by `examples_for`; the
    #: template caption keeps only the paragraphs they license.
    inputs: set[str] | None = None
    series: tmp.SeriesFacts | None = None
    tile: Tile | None = None
    #: Which prompt templates to draw from: 'train', 'eval' or 'all'.
    partition: str = 'all'

    @property
    def available(self) -> set[str]:
        """Leaf names, plus the pseudo-leaf `series`.

        `series` is not a leaf on disk. It is in the same namespace because it
        belongs to the same question -- "what is in this sample's input?" -- and a
        temporal family is unanswerable without a stack for exactly the reason a
        thermal family is unanswerable without a thermal band.
        """
        av = set(self.sheet.available)
        if self.series is not None and self.series.n_frames > 1:
            av.add('series')
        return av

    @property
    def objects(self) -> list[dict]:
        return [m for m in self.sheet.objects if m.get('bbox')]

    @property
    def spectral(self) -> dict:
        return self.sheet.raw.get('s2_spectral') or {}

    def worldcover(self) -> dict[int, float] | None:
        wc = self.sheet.raw.get('worldcover')
        if not wc:
            return None
        hist = {int(k): v for k, v in wc.items()}
        tot = sum(hist.values()) or 1
        return {k: v / tot for k, v in hist.items()}


def tile_facts(tile: Tile | str, caption: str | None = None,
               with_series: bool = True, partition: str = 'all') -> TileFacts:
    """Read everything this tile can answer from, once.

    `with_series` reads the whole stack -- twelve 13-band frames for a monthly tile
    -- and is the expensive part of this call. The arrays are cached on the `Tile`,
    so the fact sheet, the series and every family share one read.
    """
    tile = fsheet.as_tile(tile)
    sf = None
    if with_series:
        try:
            sf = tmp.series_facts(tile)
        except Exception:
            sf = None
    return TileFacts(cell=tile.cell, sheet=fsheet.fact_sheet(tile), caption=caption,
                     series=sf, tile=tile, partition=partition)


def _ask(family: str, rng: random.Random, partition: str, shape: Shape,
         **slots) -> str:
    idx = template_indices(family, partition)
    body = PROMPTS[family][idx[rng.randrange(len(idx))]].format(**slots)
    return f'{body} {FORMAT_INSTRUCTIONS[shape]}'


# --------------------------------------------------------------------------------
# Families. Each is one question shape over one measurement.
# --------------------------------------------------------------------------------
@register('caption', Shape.PROSE, (), chain_stage=3)
def _caption(t: TileFacts, rng: random.Random) -> Example | None:
    """The caller's stored caption if there is one, else the rule-based caption
    (`caption.py`), reworded each epoch and cut to the paragraphs the input licenses."""
    q = _ask('caption', rng, t.partition, Shape.PROSE)
    if t.caption:
        return Example('caption', q, t.caption, Shape.PROSE)
    text = capt.caption(t.sheet, seed=rng.randrange(2 ** 32), available=t.inputs)
    return Example('caption', q, text, Shape.PROSE) if text else None


@register('dominant_cover', Shape.WORD, ('lc',), chain_stage=3)
def _dominant_cover(t: TileFacts, rng: random.Random) -> Example | None:
    frac = t.worldcover()
    if not frac:
        return None
    top = max(frac, key=frac.get)
    truth = fsheet.WC.get(top)
    if not truth or frac[top] < 0.35:
        return None                  # no dominant class is not a dominant class
    answer, options = option_list(truth, rng, list(fsheet.WC.values()))
    q = _ask('dominant_cover', rng, t.partition, Shape.WORD)
    return Example('dominant_cover', f'{q} Options: {"; ".join(options)}.',
                   answer, Shape.WORD, {'options': options})


@register('which_present', Shape.CHOICE, (), chain_stage=3)
def _which_present(t: TileFacts, rng: random.Random) -> Example | None:
    """Forced choice, NOT yes/no.

    The prior suite retired `presence` for exactly this: a yes/no question discards the
    location it was asked about and is answerable from a prior, and forced choice
    moved the same signal from 0.5402 to 0.9425.
    """
    objs = t.objects
    if not objs:
        return None
    truth = objs[rng.randrange(min(3, len(objs)))]['name']
    decoys = [o['name'] for o in DECOY_POOL if o['name'] != truth]
    rng.shuffle(decoys)
    options = [truth] + decoys[:3]
    rng.shuffle(options)
    q = _ask('which_present', rng, t.partition, Shape.CHOICE)
    return Example('which_present', f'{q} Options: {"; ".join(options)}.',
                   truth, Shape.CHOICE, {'options': options})


#: Plausible but geographically remote names, so a decoy is never accidentally true.
DECOY_POOL = [{'name': n} for n in (
    'Puerto Miranda Ferry Terminal', 'Kalgoorlie Consolidated Gold Mines',
    'Vestfjorden Bridge', 'Rajaji Tiger Reserve', 'Lake Pukaki Canal',
    'Zaragoza Delicias Station', 'Tanjung Priok Container Terminal')]


@register('locate_point', Shape.POINT, (), chain_stage=2)
def _locate_point(t: TileFacts, rng: random.Random) -> Example | None:
    objs = [m for m in t.objects if (m['area'] or 0) > 0]
    if not objs:
        return None
    m = objs[rng.randrange(min(5, len(objs)))]
    x0, y0, x1, y1 = m['bbox']
    return Example('locate_point',
                   _ask('locate_point', rng, t.partition, Shape.POINT, label=m['name']),
                   point_target((x0 + x1) / 2, (y0 + y1) / 2, m['name']),
                   Shape.POINT, {'name': m['name'],
                                 'bbox_2d': [q1000(v) for v in m['bbox']]})


@register('locate_bbox', Shape.BBOX, (), chain_stage=2)
def _locate_bbox(t: TileFacts, rng: random.Random) -> Example | None:
    objs = [m for m in t.objects if (m['area'] or 0) > 0]
    if not objs:
        return None
    m = objs[rng.randrange(min(5, len(objs)))]
    return Example('locate_bbox',
                   _ask('locate_bbox', rng, t.partition, Shape.BBOX, label=m['name']),
                   bbox_target(m['bbox'], m['name']), Shape.BBOX,
                   {'name': m['name']})


@register('features_all', Shape.POINTS_MULTI, (), chain_stage=1)
def _features_all(t: TileFacts, rng: random.Random) -> Example | None:
    objs = [m for m in t.objects if (m['area'] or 0) > 0][:8]
    if len(objs) < 3:
        return None
    pts = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in (m['bbox'] for m in objs)]
    q = _ask('features_all', rng, t.partition, Shape.POINTS_MULTI)
    listed = '; '.join(m['name'] for m in objs)
    return Example('features_all', f'{q} The named features are: {listed}.',
                   points_multi_target(pts, 'named features'), Shape.POINTS_MULTI)


def _grid_side(rng: random.Random) -> int:
    side = rng.randint(GRID_MIN_SIDE, GRID_MAX_SIDE)
    while side * side > GRID_MAX_CELLS:
        side -= 1
    return side


@register('ndvi_grid', Shape.GRID, ('s2',), chain_stage=1)
def _ndvi_grid(t: TileFacts, rng: random.Random) -> Example | None:
    """NDVI x100 per cell, computed AT THE ASKED RESOLUTION.

    Not resampled from the fact sheet's 3x3. Upsampling a 3x3 into a 9x9 produces a
    target whose values repeat in blocks, and a model trained on that learns that
    vegetation is piecewise constant on a lattice it cannot see -- the grid would be
    teaching the interpolator, not the ground. The S2 cube is already read for the
    fact sheet, so computing it again at the requested side costs little.

    Cloud-masked, for the reason the fact sheet is: an index over cloud is an index
    of cloud, and a cell that is mostly cloud reports null rather than a plausible
    number nobody can check.
    """
    raw = t.tile.frame('s2', 0) if t.tile is not None else None
    if raw is None:
        return None
    cube = raw[[fsheet.S2['red'], fsheet.S2['nir']]].astype(np.float32) * t.tile.scale('s2')
    red, nir = cube[0], cube[1]
    ok = np.isfinite(red) & np.isfinite(nir) & (red > 0)
    cmask = fsheet.clear_mask(t.tile, 0)
    if cmask is not None and cmask.shape == ok.shape:
        ok &= cmask
    if ok.sum() < 1000:
        return None
    with np.errstate(invalid='ignore', divide='ignore'):
        ndvi = np.where(ok & (np.abs(nir + red) > 1e-6), (nir - red) / (nir + red),
                        np.nan)

    side = _grid_side(rng)
    h, w = ndvi.shape
    grid = []
    for i in range(side):
        row = []
        for j in range(side):
            blk = ndvi[i * h // side:(i + 1) * h // side,
                       j * w // side:(j + 1) * w // side]
            finite = np.isfinite(blk)
            # A cell that is mostly cloud or nodata has no honest value. Same
            # threshold as the categorical grids, and stated in the prompt.
            row.append(int(round(100 * float(np.nanmean(blk))))
                       if finite.mean() >= DEFAULT_CELL_THRESHOLD else GRID_UNKNOWN)
        grid.append(row)
    if all(v is None for r in grid for v in r):
        return None
    q = _ask('ndvi_grid', rng, t.partition, Shape.GRID)
    q += (f' Use a {side} by {side} grid and report NDVI multiplied by 100, rounded '
          f'to a whole number, as the mean over each cell. Use null where less than '
          f'{DEFAULT_CELL_THRESHOLD:.0%} of a cell is cloud-free.')
    return Example('ndvi_grid', q, grid_target('ndvi_x100', grid), Shape.GRID,
                   {'side': side, 'threshold': DEFAULT_CELL_THRESHOLD})


@register('cover_grid', Shape.GRID, ('lc',), chain_stage=1)
def _cover_grid(t: TileFacts, rng: random.Random) -> Example | None:
    """Dominant WorldCover class per cell, read from the `lc` leaf.

    Contract fix 2: the prompt STATES the threshold. A cell is labelled with a class only
    where that class covers at least half of it; otherwise the cell is null. Without
    that sentence the model is guessing which aggregation rule was used, and two
    reasonable readers of the same grid disagree.
    """
    lc = t.tile.read('lc') if t.tile is not None else None
    if lc is None:
        return None
    classes = sorted(fsheet.WC)
    legend = {c: i for i, c in enumerate(classes)}
    side = _grid_side(rng)
    h, w = lc.shape
    grid = []
    for i in range(side):
        row = []
        for j in range(side):
            blk = lc[i * h // side:(i + 1) * h // side,
                     j * w // side:(j + 1) * w // side]
            if blk.size == 0:
                row.append(GRID_UNKNOWN)
                continue
            vals, counts = np.unique(blk, return_counts=True)
            k = int(vals[counts.argmax()])
            share = counts.max() / blk.size
            row.append(legend[k] if (share >= DEFAULT_CELL_THRESHOLD
                                     and k in legend) else GRID_UNKNOWN)
        grid.append(row)
    q = _ask('cover_grid', rng, t.partition, Shape.GRID)
    q += (f' Use a {side} by {side} grid. Label a cell with a class only when that '
          f'class covers at least {DEFAULT_CELL_THRESHOLD:.0%} of the cell; '
          f'otherwise use null. Legend: '
          + '; '.join(f'{i}={fsheet.WC[c]}' for c, i in legend.items()) + '.')
    return Example('cover_grid', q, grid_target('land cover', grid), Shape.GRID,
                   {'side': side, 'threshold': DEFAULT_CELL_THRESHOLD})


@register('vegetated_fraction', Shape.DERIVED, ('s2',), chain_stage=2,
          derived_from=(Shape.NUMBER_ARRAY, Shape.NUMBER))
def _vegetated_fraction(t: TileFacts, rng: random.Random) -> Example | None:
    """Evidence then answer: the per-band greenness, then the total.

    A lone percentage is the shape the prior suite watched stall across four corpus
    generations. Making it derivable from three numbers the model writes first is
    what made that family learnable, and it also makes the answer checkable against
    the model's own output rather than only against ground truth.
    """
    sp = t.spectral
    g = sp.get('ndvi_grid')
    if 'frac_veg' not in sp or g is None or not np.isfinite(g).any():
        return None
    bands = [int(round(100 * np.nanmean(row))) if np.isfinite(row).any() else None
             for row in g]
    q = _ask('vegetated_fraction', rng, t.partition, Shape.NUMBER)
    q = (q.replace(FORMAT_INSTRUCTIONS[Shape.NUMBER], '')
         + 'First give the mean NDVI multiplied by 100 for the northern, central and '
           'southern thirds of the tile, then the vegetated percentage of the whole '
           'tile. Answer only JSON: {"ndvi_x100":[n,c,s],"vegetated_percent":N}.')
    return Example('vegetated_fraction', q.strip(),
                   derived_target('ndvi_x100', bands, 'vegetated_percent',
                                  round(100 * sp['frac_veg'], 1)),
                   Shape.DERIVED, {'evidence': Shape.NUMBER_ARRAY.value,
                                   'answer': Shape.NUMBER.value})


@register('acquisition', Shape.RECORD, (), chain_stage=3)
def _acquisition(t: TileFacts, rng: random.Random) -> Example | None:
    acq = t.sheet.raw.get('acq') or {}
    if not acq.get('date'):
        return None
    q = _ask('acquisition', rng, t.partition, Shape.RECORD)
    q += ' Use the keys "date" (YYYY-MM-DD) and "solar_elevation_deg".'
    return Example('acquisition', q,
                   record_target(date=acq['date'],
                                 solar_elevation_deg=round(acq['solar_elev'])
                                 if acq.get('solar_elev') is not None else None),
                   Shape.RECORD)


@register('relief', Shape.NUMBER, ('dem',), chain_stage=3)
def _relief(t: TileFacts, rng: random.Random) -> Example | None:
    dem = t.sheet.raw.get('dem')
    if not dem:
        return None
    q = _ask('relief', rng, t.partition, Shape.NUMBER)
    q += ' Give the difference in metres between the highest and lowest ground.'
    return Example('relief', q, scalar_target(round(dem['max'] - dem['min'])),
                   Shape.NUMBER)


# --------------------------------------------------------------------------------
# Temporal families. All require the pseudo-leaf `series`, so a monotemporal sample
# never draws one, and all read `TileFacts.series` rather than re-deriving anything.
#
# Each carries its OWN internal gate as well, because `series` only says a stack
# exists. A peak-greenness question over open water, or a growing-season count over a
# 25-day burst, is a question with no answer -- and the module that produced these
# facts had to be taught that twice.
# --------------------------------------------------------------------------------
def _usable_ndvi(sf) -> list[tuple[int, float]]:
    return [(i, sf.ndvi[i]) for i in sf.usable() if sf.ndvi[i] is not None]


@register('peak_greenness', Shape.CHOICE, ('s2', 'series'), chain_stage=3)
def _peak_greenness(t: TileFacts, rng: random.Random) -> Example | None:
    sf = t.series
    if sf is None or sf.kind is not tmp.SeriesKind.CLIMATOLOGY:
        return None                      # a 25-day burst has no "month of the year"
    if not sf.has_vegetation():
        return None                      # no peak of something that is not there
    nd = _usable_ndvi(sf)
    if len(nd) < 5:
        return None
    vals = [v for _, v in nd]
    if max(vals) - min(vals) < tmp.NDVI_AMPLITUDE_MIN:
        return None                      # an evergreen tile has no peak month
    best = max(nd, key=lambda p: p[1])[0]
    truth = sf.month_of(best)
    # Options are the months the SERIES actually contains. Offering months that were
    # never observed would make the answer findable by elimination.
    months = list(dict.fromkeys(sf.month_of(i) for i in sf.usable()))
    if truth not in months or len(months) < 3:
        return None
    rng.shuffle(months)
    options = months[:6] if truth in months[:6] else [truth] + months[:5]
    rng.shuffle(options)
    q = _ask('peak_greenness', rng, t.partition, Shape.CHOICE)
    return Example('peak_greenness', f'{q} Options: {"; ".join(options)}.',
                   truth, Shape.CHOICE, {'kind': sf.kind.value, 'options': options})


@register('green_up_count', Shape.CHOICE, ('s2', 'series'), chain_stage=3)
def _green_up_count(t: TileFacts, rng: random.Random) -> Example | None:
    """One growing season or two. The label this whole temporal effort exists for.

    CHOICE, not NUMBER, and the difference is not cosmetic: the answer set is {1, 2,
    3}, a forced choice scored 0.9916 at base where a bare scalar is the shape the
    prior suite watched stall across four corpus generations, and "2" as a number invites a
    regression metric on a categorical fact.
    """
    sf = t.series
    if sf is None or sf.kind is not tmp.SeriesKind.CLIMATOLOGY:
        return None
    if not sf.has_vegetation():
        return None
    nd = _usable_ndvi(sf)
    # A cycle count off a half-empty climatology is a count of the gaps.
    if len(nd) < 9:
        return None
    n = tmp.green_up_cycles([v for _, v in nd])
    if n < 1 or n > 3:
        return None
    truth = {1: 'one', 2: 'two', 3: 'three'}[n]
    q = _ask('green_up_count', rng, t.partition, Shape.CHOICE)
    return Example('green_up_count',
                   f'{q} Options: one; two; three. Answer with the word.',
                   truth, Shape.CHOICE, {'cycles': n,
                                         'options': ['one', 'two', 'three']})


@register('clearest_frame', Shape.CHOICE, ('s2', 'series'), chain_stage=2)
def _clearest_frame(t: TileFacts, rng: random.Random) -> Example | None:
    sf = t.series
    if sf is None or sf.n_frames < 3:
        return None
    best = max(range(sf.n_frames), key=lambda i: sf.clear[i])
    # A tie is not a question. Require the winner to be clearly the winner.
    rest = sorted((sf.clear[i] for i in range(sf.n_frames) if i != best),
                  reverse=True)
    if not rest or sf.clear[best] - rest[0] < 0.05:
        return None
    # A climatology holds two July frames, so month names are not unique and the
    # question would have two correct answers. Fall back to the full date, which
    # always is unique -- rather than dropping the family, which is what the first
    # version did and which cost every monthly tile this task.
    labels = [sf.label_of(i) for i in range(sf.n_frames)]
    if len(set(labels)) != len(labels):
        labels = [sf.day_of(i) for i in range(sf.n_frames)]
    if len(set(labels)) != len(labels):
        return None
    q = _ask('clearest_frame', rng, t.partition, Shape.CHOICE)
    return Example('clearest_frame', f'{q} Options: {"; ".join(labels)}.',
                   labels[best], Shape.CHOICE,
                   {'clear': round(sf.clear[best], 3), 'options': labels})


@register('ndvi_series', Shape.NUMBER_ARRAY, ('s2', 'series'), chain_stage=1)
def _ndvi_series(t: TileFacts, rng: random.Random) -> Example | None:
    sf = t.series
    if sf is None or sf.n_frames < 3:
        return None
    vals = [None if v is None else int(round(100 * v)) for v in sf.ndvi]
    if sum(v is not None for v in vals) < 3:
        return None
    q = _ask('ndvi_series', rng, t.partition, Shape.NUMBER_ARRAY)
    q += (f' There are {sf.n_frames} acquisitions, in order: '
          f'{", ".join(sf.label_of(i) for i in range(sf.n_frames))}. Give NDVI '
          f'multiplied by 100 and rounded to a whole number for each, as a JSON '
          f'array, using null for an acquisition too cloudy to measure.')
    return Example('ndvi_series', q, scalar_target(vals), Shape.NUMBER_ARRAY)


@register('disturbance', Shape.DERIVED, ('s2', 'series'), chain_stage=2,
          derived_from=(Shape.NUMBER_ARRAY, Shape.CHOICE))
def _disturbance(t: TileFacts, rng: random.Random) -> Example | None:
    """Per-step changed AREA first, then the date of the worst step.

    Evidence before answer, and the evidence is the thing that makes the answer
    checkable: a model that names a date without the areas has guessed, and the two
    can be scored separately when it gets one right and the other wrong.

    Bursts only. In a climatology the neighbouring frame can be years away, so a
    large change between them is a season, not an event, and `temporal.py` words it
    that way rather than calling it a disturbance.
    """
    sf = t.series
    if sf is None or sf.kind is not tmp.SeriesKind.BURST:
        return None
    if not sf.dnbr_area or not sf.is_mostly_land():
        return None
    areas = [(i, round(100 * a, 1)) for i, a in sf.dnbr_area]
    worst = max(areas, key=lambda p: p[1])
    if worst[1] < 100 * tmp.DNBR_AREA_MIN:
        return None
    labels = [sf.label_of(i) for i, _ in areas]
    if len(set(labels)) != len(labels):
        labels = [sf.day_of(i) for i, _ in areas]
    if len(set(labels)) != len(labels):
        return None
    # The CHOICE boilerplate is stripped, not appended to: "answer with one option
    # copied exactly, and nothing else" directly contradicts the JSON object this
    # family wants, and a prompt that states two formats gets neither.
    q = _ask('disturbance', rng, t.partition, Shape.CHOICE).replace(
        FORMAT_INSTRUCTIONS[Shape.CHOICE], '').strip()
    q += (' First give, for each acquisition after the first readable one, the '
          'percentage of the tile whose surface changed abruptly since the previous '
          'one; then name the date with the largest change. Answer only JSON: '
          '{"changed_percent":[...],"date":"..."}. The acquisitions in order are: '
          + '; '.join(labels) + '.')
    return Example('disturbance', q,
                   derived_target('changed_percent', [a for _, a in areas],
                                  'date', labels[areas.index(worst)]),
                   Shape.DERIVED, {'max_area_pct': worst[1], 'options': labels,
                                   'evidence': Shape.NUMBER_ARRAY.value,
                                   'answer': Shape.CHOICE.value})


# --------------------------------------------------------------------------------
# Sampling and conversation packing.
# --------------------------------------------------------------------------------
def examples_for(tile: Tile | str | None = None, *, available: set[str] | None = None,
                 caption: str | None = None, epoch: int = 0, seed: int = 0,
                 max_feasibility: Feasibility | None = None,
                 facts: TileFacts | None = None,
                 partition: str = 'all') -> list[Example]:
    """Every task this tile and this encoder configuration license, this epoch.

    `available` simulates a narrower encoder input than the cell actually has, and a
    family whose `requires` is unmet is DROPPED -- never rewritten -- for the same
    reason the caption's paragraphs are: a question about a band that is not in the
    input is a label the input cannot support.

    `max_feasibility` filters by how well a FROZEN decoder can emit the shape, which
    is what stage-1 training needs and stage-2 training does not. Nothing is removed
    from the registry; the mixture chooses.

    `partition` picks the prompt templates: 'train', 'eval' (the held-out 20% of
    each paraphrase bank) or 'all'.
    """
    t = facts or tile_facts(tile, caption, partition=partition)
    if facts is not None:
        t.partition = partition
    cell = t.cell
    if caption and not t.caption:
        t.caption = caption
    av = set(available) if available is not None else set(t.available)
    t.inputs = av & set(t.sheet.available)
    from .shapes import FEASIBILITY_RANK
    cap = FEASIBILITY_RANK[max_feasibility] if max_feasibility else 99

    out = []
    for name, fam in FAMILIES.items():
        if not fam.requires <= av:
            continue
        if FEASIBILITY_RANK.get(fam.feasibility, 0) > cap:
            continue
        rng = random.Random(f'{seed}:{cell}:{name}:{epoch}')
        try:
            ex = fam.build(t, rng)
        except Exception:
            continue
        if ex is not None:
            out.append(ex)
    return out


def chain_order(examples: Sequence[Example]) -> list[Example]:
    """Dense spatial output first, naming last. Ties keep their incoming order."""
    return [e for _, e in sorted(
        enumerate(examples),
        key=lambda p: (FAMILIES[p[1].family].chain_stage, p[0]))]


def pack_conversation(examples: Sequence[Example], rng: random.Random, *,
                      max_turns: int = 4,
                      token_budget: int = 1800) -> list[Example] | None:
    """Fold several tasks over ONE tile into an ordered multi-turn exchange.

    One image, several questions: the encoder runs once and supervises four answers
    instead of one, and two things become supervisable that no single-turn example
    can express -- consistency (a total must agree with the grid that preceded it, a
    point must fall inside the box), which is checkable WITHOUT ground truth and so
    transfers to unlabelled imagery, and attribution (when an ordering is wrong,
    whether the per-cell estimates or only the sort was wrong).

    Returns None below two turns: a one-turn conversation is an example with extra
    machinery. The caller emits those singly.
    """
    ordered = chain_order(examples)
    kept, used = [], 0
    for ex in ordered:
        cost = FAMILIES[ex.family].spec.max_target_tokens + len(ex.question.split()) * 2
        if kept and used + cost > token_budget:
            break
        kept.append(ex)
        used += cost
        if len(kept) >= max_turns:
            break
    return kept if len(kept) >= 2 else None


def registry_table() -> list[dict]:
    """The contract, as rows -- for a report or a sanity check."""
    return [{'family': f.name, 'shape': f.shape.value,
             'frozen_llm_feasibility': f.feasibility.value,
             'max_target_tokens': f.spec.max_target_tokens,
             'requires': ', '.join(sorted(f.requires)) or '—',
             'chain_stage': f.chain_stage}
            for f in FAMILIES.values()]
