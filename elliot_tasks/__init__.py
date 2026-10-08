"""Measured fact sheets and on-the-fly vision-language tasks over ELLIOT tiles.

Reads ELLIOT-Pretrain and its extension ELLIOT-X-EXT through the TACO reader, and
builds from them a modality-tagged fact sheet, a rule-based caption, series facts for the monthly and burst
parts, sixteen task families with answer shapes, scorers for those answers, and a
PyTorch dataset that yields (arrays, conversation) pairs.
"""
from . import caption, caption_check
from .data import (
                   MODALITIES,
                   PARTS,
                   Part,
                   Roots,
                   Source,
                   Tile,
                   configure,
                   find,
                   open_part,
                   roots,
                   tile,
)
from .factsheet import SEGMENTS, FactSheet, assemble, build_user_prompt, fact_sheet
from .factstore import configure_facts, facts
from .scoring import Score, score, score_conversation
from .shapes import SHAPES, Feasibility, Shape
from .tasks import (
                   FAMILIES,
                   Example,
                   TileFacts,
                   chain_order,
                   examples_for,
                   pack_conversation,
                   registry_table,
                   tile_facts,
)
from .temporal import SeriesFacts, SeriesKind, series_facts, temporal_block

__version__ = '0.1.0'

__all__ = [
                   'FAMILIES',
                   'MODALITIES',
                   'PARTS',
                   'SEGMENTS',
                   'SHAPES',
                   'Example',
                   'FactSheet',
                   'Feasibility',
                   'Part',
                   'Roots',
                   'Score',
                   'SeriesFacts',
                   'SeriesKind',
                   'Shape',
                   'Source',
                   'Tile',
                   'TileFacts',
                   'assemble',
                   'build_user_prompt',
                   'caption',
                   'caption_check',
                   'chain_order',
                   'configure',
                   'configure_facts',
                   'examples_for',
                   'fact_sheet',
                   'facts',
                   'find',
                   'open_part',
                   'pack_conversation',
                   'registry_table',
                   'roots',
                   'score',
                   'score_conversation',
                   'series_facts',
                   'temporal_block',
                   'tile',
                   'tile_facts',
]
