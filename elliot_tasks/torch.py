"""A thin PyTorch `Dataset` over ELLIOT tiles: (arrays, conversation) per item.

Each item is one tile of one part. The arrays are the modalities named in
`available`, read once through the TACO reader; the conversation is a list of chat
turns built on the fly from the tasks those same modalities license. Removing a
modality removes both its array AND every question about it, so the loader never asks
about a sensor it did not supply.

Randomness is a pure function of `(seed, cell, epoch)`: the same tile in the same
epoch gives the same conversation in any worker and on any machine, and a new epoch
resamples the wording and the grid sizes. Call `set_epoch` from the training loop.

Arrays, as float32 tensors with nodata as NaN unless stated:

  s2    (T, 13, H, W)  top-of-atmosphere reflectance (stored value x 1e-4)
  l8    (T, 11, H, W)  reflectance; bands 9-10 brightness temperature in kelvin
  s1    (T, 2, H, W)   VV, VH gamma-0, linear power
  dem   (H, W)         metres
  lc    (H, W)         int64 ESA WorldCover codes (0 = no data)
  cloud_mask_s2, cloud_mask_l8  (T, H, W) int64: 0 clear, 1 thick, 2 thin, 3 shadow

T is 1 for monotemporal, 12 for monthly and 6 for burst.
"""
from __future__ import annotations

import random
from collections.abc import Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from . import tasks as T
from .data import MODALITIES, SLOTS, open_part
from .shapes import Feasibility

FLOAT_NODATA = {'s2': 0, 'l8': 0, 's1': 0}


def _tensor(tile, key: str) -> torch.Tensor | None:
    arr = tile.read(key)
    if arr is None:
        return None
    if key in ('lc',) or key.startswith('cloud_mask'):
        return torch.from_numpy(np.ascontiguousarray(arr).astype(np.int64))
    out = arr.astype(np.float32) * np.float32(tile.scale(key))
    if key in FLOAT_NODATA:
        out[arr == FLOAT_NODATA[key]] = np.nan
    return torch.from_numpy(np.ascontiguousarray(out))


def to_chat(examples: Sequence[T.Example]) -> list[dict]:
    """Examples as alternating user / assistant turns."""
    turns = []
    for e in examples:
        turns.append({'role': 'user', 'content': e.question})
        turns.append({'role': 'assistant', 'content': e.target})
    return turns


class ElliotTaskDataset(Dataset):
    """ELLIOT tiles as (arrays, conversation) pairs.

    part         'monotemporal', 'monthly' or 'burst'
    indices      which rows of that part (default: all)
    available    modalities to load and to ask about: any of 's2', 'l8', 's1',
                 'dem', 'lc'; add 'cloud_mask_s2' / 'cloud_mask_l8' to also load
                 the masks (they license no task of their own)
    mode         'conversation': up to `max_turns` tasks packed into one exchange
                 (a single turn when only one task applies); 'all': every task the
                 tile licenses, one turn each, in chain order
    partition    prompt templates: 'train', 'eval' or 'all'
    captions     optional {cell: caption} for the `caption` family
    """

    def __init__(self, part: str, indices: Iterable[int] | None = None, *,
                 available: Iterable[str] = MODALITIES, mode: str = 'conversation',
                 epoch: int = 0, seed: int = 0, partition: str = 'all',
                 max_turns: int = 4, token_budget: int = 1800,
                 max_feasibility: Feasibility | None = None,
                 captions: dict[str, str] | None = None):
        if mode not in ('conversation', 'all'):
            raise ValueError("mode must be 'conversation' or 'all'")
        self.available = tuple(available)
        if unknown := set(self.available) - set(SLOTS):
            raise ValueError(f'unknown modalities {sorted(unknown)}; have {sorted(SLOTS)}')
        self.part_name = part
        self.indices = list(range(len(open_part(part)))) if indices is None \
            else [int(i) for i in indices]
        self.mode, self.epoch, self.seed = mode, epoch, seed
        self.partition, self.max_turns, self.token_budget = partition, max_turns, token_budget
        self.max_feasibility = max_feasibility
        self.captions = captions or {}

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.indices)

    def tile(self, i: int):
        return open_part(self.part_name).tile(self.indices[i])

    def __getitem__(self, i: int) -> tuple[dict[str, torch.Tensor], list[dict]]:
        tile = self.tile(i)
        # The fact sheet only states what the encoder is given: `series` is kept
        # because it is a property of the stack, not a sensor.
        task_mods = {m for m in self.available if m in MODALITIES}
        facts = T.tile_facts(tile, self.captions.get(tile.cell),
                             with_series='s2' in task_mods, partition=self.partition)
        av = (facts.available & task_mods) | ({'series'} & facts.available)
        examples = T.examples_for(facts=facts, available=av, epoch=self.epoch,
                                  seed=self.seed, max_feasibility=self.max_feasibility,
                                  partition=self.partition)
        if self.mode == 'all':
            turns = T.chain_order(examples)
        else:
            rng = random.Random(f'{self.seed}:{tile.cell}:conversation:{self.epoch}')
            # Draw which tasks this conversation holds, then let `pack_conversation`
            # order them (dense spatial output first) and enforce the token budget.
            pool = list(examples)
            rng.shuffle(pool)
            pool = pool[:self.max_turns]
            turns = T.pack_conversation(pool, rng, max_turns=self.max_turns,
                                        token_budget=self.token_budget) \
                or T.chain_order(pool)[:1]
        arrays = {k: t for k in self.available if (t := _tensor(tile, k)) is not None}
        tile.drop_arrays()
        return arrays, to_chat(turns)


def collate(batch: list[tuple[dict, list]]) -> tuple[list[dict], list[list]]:
    """Keep a batch as lists: tiles differ in which modalities they carry and
    conversations differ in length, so stacking is the model's decision."""
    return [b[0] for b in batch], [b[1] for b in batch]
