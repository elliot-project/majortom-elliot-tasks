"""The answer SHAPES a decoder is asked to emit, and how feasible each is frozen.

WHY SHAPE AND NOT FAMILY. In stage-1 training the language model's weights do not
move: the gradient reaches only the adapter between the EO encoder and the decoder.
The decoder must therefore already put probability mass on the target's FORMAT, or
the adapter is being asked to fix the decoder's grammar instead of to encode the
image. Whether it can is a property of what is being emitted -- `{"bbox_2d":[...]}`,
a 64-cell integer grid, one word copied from a list -- and two task families that
emit the same thing behave identically. So the flag lives here, on thirteen shapes,
and families inherit it.

THIS IS A TRAINING CONCERN, NOT A DATA ONE: a dataset describes what was measured,
a mixture describes how it is taught.

`max_target_tokens` is the mechanism, not decoration. A 64-cell grid is roughly 380
target tokens, of which the brackets, commas and the echoed `shape` are predictable
from the format alone, so the share of the cross-entropy that actually depends on the
adapter's embedding is small. Calling that "medium feasibility" is an adjective;
carrying the token budget lets the sampler weight by it.

EVIDENCE. Every `evidence` string is either a measurement or says it is not. Two
sources: a text-only probe of a frozen decoder (Qwen3-VL-2B's language model) asked
to emit each shape when handed the answer ("oracle"), when handed the fact sheet
("told"), and with neither ("blind"); and base-model scores of a VLM reading a real
image on an earlier task suite ("prior suite"), which is an easier setting than an
adapter's embedding for CONTENT and roughly fair for FORMAT.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Shape(str, Enum):
    """What the decoder emits. Thirteen, plus the DERIVED wrapper below."""
    PROSE = 'prose'                  # free text, no schema
    WORD = 'word'                    # one name from an open vocabulary
    CHOICE = 'choice'                # one item copied from a list in the prompt
    NUMBER = 'number'                # one bare number, unit stated in the prompt
    NUMBER_ARRAY = 'number_array'    # [0.31,0.44,...] -- one per band or per frame
    ORDER = 'order'                  # a permutation
    POINT = 'point'                  # {"point_2d":[x,y],"label":...}
    BBOX = 'bbox'                    # {"bbox_2d":[x0,y0,x1,y1],"label":...}
    POINTS_MULTI = 'points_multi'    # variable-length point set
    BOXES_MULTI = 'boxes_multi'      # variable-length box set
    VERTICES = 'vertices'            # polygon / interior-seed vertex list
    GRID = 'grid'                    # {"variable":...,"shape":[r,c],"grid":[[ints]]}
    RECORD = 'record'                # sorted-key JSON of a few scalars

    #: NOT a shape. A wrapper that puts evidence before an answer in one object, so
    #: a model that writes the conclusion first has visibly not derived it. It
    #: composes with any shape above; see `derived_spec`.
    DERIVED = 'derived'


class Feasibility(str, Enum):
    """How well a FROZEN decoder handles this shape.

    Named for what it measures rather than for a training stage: `s1`/`s2` would
    collide with Sentinel-1 and Sentinel-2, which appear as modality keys three
    modules away in this same package.
    """
    HIGH = 'high'       # emits and scores it frozen; weight freely in stage 1
    MEDIUM = 'medium'   # format survives, signal is diluted or calibration is off
    LOW = 'low'         # needs a tuned decoder to be SCOREABLE, not merely good


@dataclass(frozen=True)
class ShapeSpec:
    shape: Shape
    feasibility: Feasibility
    max_target_tokens: int
    evidence: str


SHAPES: dict[Shape, ShapeSpec] = {
    Shape.PROSE: ShapeSpec(
        Shape.PROSE, Feasibility.HIGH, 180,
        'MEASURED wellformed 1.00 at a median 136 tokens; no schema to violate, and '
        'the classic stage-1 shape'),
    Shape.WORD: ShapeSpec(
        Shape.WORD, Feasibility.HIGH, 8,
        'MEASURED oracle 1.00, told 1.00, blind 0.27; prior suite base osm_scene_class '
        '0.6207 -- format was never the constraint'),
    Shape.CHOICE: ShapeSpec(
        Shape.CHOICE, Feasibility.HIGH, 12,
        'MEASURED oracle 1.00, told 1.00, blind 0.00; prior suite base option_adherence '
        '0.9916'),
    Shape.NUMBER: ShapeSpec(
        Shape.NUMBER, Feasibility.MEDIUM, 8,
        'MEASURED format 1.00 oracle/1.00 told/0.18 blind. Stays MEDIUM on '
        'CALIBRATION, which this probe cannot test: reading a number off a prose '
        'sheet is not producing one from an embedding, and prior suite population_log_mae '
        'was 1.63 at base and still 1.03 trained'),
    Shape.NUMBER_ARRAY: ShapeSpec(
        Shape.NUMBER_ARRAY, Feasibility.MEDIUM, 40,
        'MEASURED format 1.00 oracle. MEDIUM for the same calibration reason as '
        'NUMBER, multiplied by the element count'),
    Shape.ORDER: ShapeSpec(
        Shape.ORDER, Feasibility.MEDIUM, 20,
        'MEASURED oracle 0.70 -- it cannot reliably transcribe an ordering it was '
        'HANDED -- and told 0.40 against blind 0.40, i.e. NO told-over-blind gap at '
        'all. Consistent with prior suite seasonal_order_exact pinned at chance'),
    Shape.POINT: ShapeSpec(
        Shape.POINT, Feasibility.HIGH, 30,
        'MEASURED oracle 1.00 parse/wellformed/accuracy, told 0.38 from compass '
        'words alone against blind 0.12. Qwen is natively trained on the [0,1000] '
        'convention -- NOT free for a decoder that is not'),
    Shape.BBOX: ShapeSpec(
        Shape.BBOX, Feasibility.HIGH, 40,
        'MEASURED oracle 1.00, told 0.38 against blind 0.00. Same convention caveat '
        'as POINT'),
    Shape.POINTS_MULTI: ShapeSpec(
        Shape.POINTS_MULTI, Feasibility.LOW, 200,
        'LOW on CONTENT, not on format, and the correction matters: MEASURED oracle '
        '1.00 at 69 tokens, so the schema is fine when the answer is known. Told 0.00 '
        'and prior suite base instance_f1 0.0205. The fix is supervision, not a schema'),
    Shape.BOXES_MULTI: ShapeSpec(
        Shape.BOXES_MULTI, Feasibility.LOW, 320,
        'as POINTS_MULTI: MEASURED oracle 1.00 at 121 tokens. But WITHOUT the answer '
        'it ran to the 600-token cap on every probe -- an unparseable answer of '
        'maximum length, which costs the whole batch its sequence budget. prior suite '
        'shipped ZERO multi-box training targets'),
    Shape.VERTICES: ShapeSpec(
        Shape.VERTICES, Feasibility.LOW, 260,
        'MEASURED oracle parse 0.62 and accuracy 0.00 -- the ONLY shape that fails to '
        'parse even when handed its answer. The 512-token polygon answers prior suite '
        'deliberately retired'),
    Shape.GRID: ShapeSpec(
        Shape.GRID, Feasibility.LOW, 400,
        'DOWNGRADED from MEDIUM by measurement. Oracle WELLFORMED 1.00 but oracle '
        'ACCURACY 0.09 (3x3) and 0.18 (8x8): the decoder emits a perfectly shaped '
        'lattice and fills it wrongly EVEN WHEN THE ANSWER IS IN THE PROMPT. That is '
        'a transcription failure, not a perception one, and no adapter fixes it. '
        'Consistent with prior suite base grid_shape_valid 0.8861 against '
        'categorical_grid_accuracy 0.1666 -- it draws the box and cannot fill it'),
    Shape.RECORD: ShapeSpec(
        Shape.RECORD, Feasibility.HIGH, 40,
        'MEASURED oracle 1.00, told 1.00, blind 0.00'),
}

#: Order matters and is the reasoning order; `derived_target` preserves it.
FEASIBILITY_RANK = {Feasibility.HIGH: 0, Feasibility.MEDIUM: 1, Feasibility.LOW: 2}


def derived_spec(evidence: Shape, answer: Shape) -> ShapeSpec:
    """The spec for `{evidence: ..., answer: ...}` wrapping two shapes.

    Two rules, and the second is not symmetric with the first.

    1. The wrapper inherits the STRICTER feasibility of its parts and the SUM of
       their budgets, because the decoder has to emit both.
    2. EXCEPT that it PROMOTES a bare NUMBER one step. The lesson of the prior
       suite, and the reason it introduced conversations at all, is that every
       family whose answer was a lone scalar stalled across four corpus generations, while the ones with internal
       structure learned; wrapping the scalar so it is derivable from evidence the
       model itself just wrote is what made those families learnable. So
       `derived[GRID -> NUMBER]` is a better frozen-decoder citizen than `NUMBER`
       alone, despite being longer -- which is exactly the case where the token
       budget and the feasibility flag disagree, and the flag wins.
    """
    e, a = SHAPES[evidence], SHAPES[answer]
    rank = max(FEASIBILITY_RANK[e.feasibility], FEASIBILITY_RANK[a.feasibility])
    if answer is Shape.NUMBER and e.feasibility is not Feasibility.LOW:
        rank = max(0, rank - 1)
    feas = next(f for f, r in FEASIBILITY_RANK.items() if r == rank)
    return ShapeSpec(Shape.DERIVED, feas,
                     e.max_target_tokens + a.max_target_tokens,
                     f'derived[{evidence.value} -> {answer.value}]: {a.evidence}')
