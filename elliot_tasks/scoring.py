"""Parse and score a model's answer to a task, one rule per answer shape.

Tasks and scorers ship together: every family's target is written by `tasks.py` in
one of the shapes of `shapes.py`, and every shape has a parser and a scorer here.
Each scorer returns a `Score` with three separate readings, because they fail for
different reasons and are fixed by different means:

  parsed       the answer could be read at all (JSON found, a number found, ...)
  wellformed   it has the declared form (the right keys, a grid of the asked size,
               integers on [0, 1000], an option copied from the list)
  score        how right it is, on [0, 1] (None where nothing can be compared)

An answer that does not parse scores 0, not "skipped": failing to answer in the
declared shape is a wrong answer.

Tolerances are stated per quantity in `TOLERANCE` and are deliberately loose enough to
forgive rounding and tight enough to separate a reading from a guess. Coordinates
are integers on the [0, 1000] grid the prompts use.
"""
from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .shapes import Shape

#: (absolute, relative) tolerance per quantity: a value is correct when its error is
#: within EITHER. Keyed by family name or by the JSON key the value sits under.
TOLERANCE: dict[str, tuple[float, float]] = {
    'relief': (10.0, 0.10),                # metres
    'vegetated_percent': (10.0, 0.0),      # percentage points
    'ndvi_x100': (5.0, 0.0),               # NDVI x100, i.e. 0.05 NDVI
    'ndvi_series': (5.0, 0.0),
    'ndvi_grid': (5.0, 0.0),
    'changed_percent': (2.0, 0.0),         # percentage points of the tile
    'solar_elevation_deg': (2.0, 0.0),     # degrees
}
DEFAULT_TOLERANCE = (0.0, 0.0)

#: A point is a hit inside the reference object's box when it has one, otherwise
#: within this many grid units of the reference point on both axes. Also the
#: matching radius for multi-point answers.
POINT_RADIUS = 100
BBOX_IOU_HIT = 0.5


@dataclass
class Score:
    parsed: bool
    wellformed: bool
    score: float | None
    details: dict = field(default_factory=dict)

    @property
    def correct(self) -> bool:
        return self.score is not None and self.score >= 1.0 - 1e-9


# --------------------------------------------------------------------------------
# Parsing.
# --------------------------------------------------------------------------------
def normalize_text(value: Any) -> str:
    """Unicode, case and punctuation folded, whitespace collapsed."""
    value = unicodedata.normalize('NFKC', str(value)).casefold()
    value = ''.join(' ' if unicodedata.category(c)[0] in 'PS' else c for c in value)
    return re.sub(r'\s+', ' ', value).strip()


def parse_json(text: str) -> Any:
    """The first JSON object or array in `text`, tolerating code fences and chatter."""
    t = (text or '').strip()
    if t.startswith('```'):
        t = t.split('\n', 1)[-1].rsplit('```', 1)[0]
    try:
        return json.loads(t)
    except (json.JSONDecodeError, ValueError):
        pass
    for lo, hi in (('{', '}'), ('[', ']')):
        i, j = t.find(lo), t.rfind(hi)
        if 0 <= i < j:
            try:
                return json.loads(t[i:j + 1])
            except json.JSONDecodeError:
                continue
    return None


_NUMBER = re.compile(r'-?\d+(?:\.\d+)?')


def parse_number(text: str) -> float | None:
    m = _NUMBER.search(text or '')
    return float(m.group()) if m else None


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _ints(v: Any, n: int | None = None) -> bool:
    return (isinstance(v, list) and all(isinstance(x, int) and not isinstance(x, bool)
                                        for x in v) and (n is None or len(v) == n))


def _on_grid(v: Sequence[int]) -> bool:
    return all(0 <= x <= 1000 for x in v)


def _close(got: float, want: float, tol: tuple[float, float]) -> bool:
    err = abs(got - want)
    return err <= tol[0] or err <= tol[1] * abs(want) or err < 1e-9


# --------------------------------------------------------------------------------
# One scorer per shape.
# --------------------------------------------------------------------------------
def token_f1(prediction: str, reference: str) -> float:
    p, r = normalize_text(prediction).split(), normalize_text(reference).split()
    if not p or not r:
        return float(p == r)
    overlap = sum((Counter(p) & Counter(r)).values())
    if not overlap:
        return 0.0
    prec, rec = overlap / len(p), overlap / len(r)
    return 2 * prec * rec / (prec + rec)


def score_prose(answer: str, target: str | None) -> Score:
    """Token F1 against the reference text. Prose has no schema to violate; it is
    well formed at 8-140 words with no JSON in it."""
    words = len((answer or '').split())
    wf = 8 <= words <= 140 and '{' not in (answer or '')
    return Score(bool(answer and answer.strip()), wf,
                 token_f1(answer, target) if target else None, {'words': words})


def score_choice(answer: str, target: str, options: Sequence[str] | None = None) -> Score:
    """Exact match after normalisation. Well formed when it is one of the options."""
    got = normalize_text((answer or '').strip().strip('."\''))
    opts = {normalize_text(o) for o in options} if options else None
    wf = bool(got) and (opts is None or got in opts)
    return Score(bool(got), wf, float(got == normalize_text(target)), {'answer': got})


def score_number(answer: str, target: float, tol: tuple[float, float]) -> Score:
    """The first number in the answer, within tolerance. Well formed when the
    answer is that number and nothing else."""
    v = parse_number(answer)
    if v is None:
        return Score(False, False, 0.0)
    wf = bool(re.fullmatch(r'\s*-?\d+(?:\.\d+)?\s*', answer or ''))
    return Score(True, wf, float(_close(v, float(target), tol)),
                 {'value': v, 'abs_error': abs(v - float(target))})


def score_number_array(got: Any, want: Sequence[Any], tol: tuple[float, float]) -> Score:
    """Element-wise, nulls included: a null target is matched only by a null.

    `score` is the share of positions answered correctly; `mae` is over the
    positions where both sides hold a number.
    """
    if not isinstance(got, list):
        return Score(got is not None, False, 0.0)
    wf = len(got) == len(want) and all(g is None or _is_number(g) for g in got)
    if not wf:
        return Score(True, False, 0.0, {'length': len(got), 'expected': len(want)})
    hits, errs = 0, []
    for g, w in zip(got, want):
        if w is None or g is None:
            hits += g is None and w is None
            continue
        errs.append(abs(g - w))
        hits += _close(g, w, tol)
    return Score(True, True, hits / len(want) if want else 1.0,
                 {'mae': sum(errs) / len(errs) if errs else None})


def score_point(value: Any, target: dict, bbox: Sequence[int] | None = None) -> Score:
    p = value.get('point_2d') if isinstance(value, dict) else None
    if not (_ints(p, 2) and _on_grid(p)):
        return Score(value is not None, False, 0.0)
    t = target['point_2d']
    dist = math.hypot(p[0] - t[0], p[1] - t[1])
    if bbox is not None:
        hit = bbox[0] <= p[0] <= bbox[2] and bbox[1] <= p[1] <= bbox[3]
    else:
        hit = abs(p[0] - t[0]) <= POINT_RADIUS and abs(p[1] - t[1]) <= POINT_RADIUS
    return Score(True, True, float(hit), {'distance': dist, 'inside_box': bbox is not None})


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def score_bbox(value: Any, target: dict) -> Score:
    """IoU with the reference box; `details['hit']` at IoU >= 0.5."""
    b = value.get('bbox_2d') if isinstance(value, dict) else None
    if not (_ints(b, 4) and _on_grid(b) and b[0] < b[2] and b[1] < b[3]):
        return Score(value is not None, False, 0.0)
    iou = box_iou(b, target['bbox_2d'])
    return Score(True, True, iou, {'iou': iou, 'hit': iou >= BBOX_IOU_HIT})


def score_points_multi(value: Any, target: dict) -> Score:
    """F1 of a one-to-one greedy match, closest pairs first, within `POINT_RADIUS`.
    Order-free: the target's order is for learnability, not for scoring."""
    pts = value.get('points_2d') if isinstance(value, dict) else None
    if not (isinstance(pts, list) and all(_ints(p, 2) and _on_grid(p) for p in pts)):
        return Score(value is not None, False, 0.0)
    want = target['points_2d']
    pairs = sorted((math.hypot(p[0] - w[0], p[1] - w[1]), i, j)
                   for i, p in enumerate(pts) for j, w in enumerate(want))
    used_p, used_w, tp = set(), set(), 0
    for d, i, j in pairs:
        if d > POINT_RADIUS:
            break
        if i in used_p or j in used_w:
            continue
        used_p.add(i); used_w.add(j); tp += 1
    prec = tp / len(pts) if pts else 0.0
    rec = tp / len(want) if want else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return Score(True, True, f1, {'precision': prec, 'recall': rec, 'matched': tp})


def score_grid(value: Any, target: dict, tol: tuple[float, float]) -> Score:
    """Cell accuracy on a grid of exactly the asked shape.

    Categorical grids (land-cover legend indices) need the exact class; numeric
    grids need the value within tolerance. A null target cell is matched only by a
    null. `mae` is over cells where both sides are numbers (numeric grids only).
    """
    if not isinstance(value, dict):
        return Score(value is not None, False, 0.0)
    rows, cols = target['shape']
    grid = value.get('grid')
    wf = (value.get('shape') == [rows, cols] and isinstance(grid, list)
          and len(grid) == rows
          and all(isinstance(r, list) and len(r) == cols for r in grid)
          and all(v is None or _is_number(v) for r in grid for v in r))
    if not wf:
        return Score(True, False, 0.0)
    categorical = target.get('variable') == 'land cover'
    hits, errs = 0, []
    for gr, tr in zip(grid, target['grid']):
        for g, t in zip(gr, tr):
            if t is None or g is None:
                hits += g is None and t is None
                continue
            if categorical:
                hits += g == t
            else:
                errs.append(abs(g - t))
                hits += _close(g, t, tol)
    return Score(True, True, hits / (rows * cols),
                 {'mae': (sum(errs) / len(errs)) if errs else None,
                  'categorical': categorical})


def score_record(value: Any, target: dict) -> Score:
    """Key by key: numbers within the key's tolerance, anything else exactly.
    `score` is the share of the target's keys answered correctly."""
    if not isinstance(value, dict):
        return Score(value is not None, False, 0.0)
    wf = set(target) <= set(value)
    per = {}
    for k, want in target.items():
        got = value.get(k)
        if want is None or got is None:
            per[k] = want is None and got is None
        elif _is_number(want) and _is_number(got):
            per[k] = _close(got, want, TOLERANCE.get(k, DEFAULT_TOLERANCE))
        else:
            per[k] = normalize_text(got) == normalize_text(want)
    return Score(True, wf, sum(per.values()) / len(per) if per else 1.0, {'keys': per})


# --------------------------------------------------------------------------------
# Dispatch.
# --------------------------------------------------------------------------------

def score(example, answer: str) -> Score:
    """Score a model's `answer` to a `tasks.Example`."""
    from .tasks import FAMILIES
    fam, shape, meta = example.family, example.shape, example.meta or {}
    tol = TOLERANCE.get(fam, DEFAULT_TOLERANCE)
    if shape is Shape.PROSE:
        return score_prose(answer, example.target)
    if shape in (Shape.WORD, Shape.CHOICE):
        return score_choice(answer, example.target, meta.get('options'))
    if shape is Shape.NUMBER:
        return score_number(answer, float(example.target), tol)

    target = json.loads(example.target)
    value = parse_json(answer)
    if value is None:
        return Score(False, False, 0.0)
    if shape is Shape.NUMBER_ARRAY:
        return score_number_array(value, target, tol)
    if shape is Shape.POINT:
        return score_point(value, target, meta.get('bbox_2d'))
    if shape is Shape.BBOX:
        return score_bbox(value, target)
    if shape is Shape.POINTS_MULTI:
        return score_points_multi(value, target)
    if shape is Shape.GRID:
        return score_grid(value, target, tol)
    if shape is Shape.RECORD:
        return score_record(value, target)
    if shape is Shape.DERIVED:
        return _score_derived(value, target, FAMILIES[fam].derived_from, meta)
    raise ValueError(f'no scorer for shape {shape}')


def _score_derived(value: Any, target: dict, shapes: tuple[Shape, Shape],
                   meta: dict) -> Score:
    """Evidence and answer scored separately; `score` is the ANSWER's.

    The evidence score is in `details` so a right answer reached from wrong
    evidence -- or the reverse -- is visible rather than averaged away.
    """
    (ek, ev), (ak, av) = list(target.items())[:2]
    if not isinstance(value, dict):
        return Score(True, False, 0.0)
    e_shape, a_shape = shapes
    wf = ek in value and ak in value
    e = (score_number_array(value.get(ek), ev, TOLERANCE.get(ek, DEFAULT_TOLERANCE))
         if e_shape is Shape.NUMBER_ARRAY else Score(True, True, None))
    got = value.get(ak)
    if got is None:
        a = Score(False, False, 0.0)
    elif a_shape is Shape.NUMBER:
        a = (Score(True, True, float(_close(got, av, TOLERANCE.get(ak, DEFAULT_TOLERANCE))),
                   {'abs_error': abs(got - av)})
             if _is_number(got) else Score(True, False, 0.0))
    else:
        a = score_choice(str(got), str(av), meta.get('options'))
    return Score(True, wf and e.wellformed and a.wellformed, a.score,
                 {'evidence_score': e.score, 'evidence': e.details,
                  'answer': a.details})


def score_conversation(examples: Sequence, answers: Sequence[str]) -> list[Score]:
    """Score each turn of a packed conversation."""
    if len(examples) != len(answers):
        raise ValueError(f'{len(examples)} turns, {len(answers)} answers')
    return [score(e, a) for e, a in zip(examples, answers)]
