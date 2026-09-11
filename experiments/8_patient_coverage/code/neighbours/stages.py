"""Per-shard fit stage for the patient-coverage experiment."""

from __future__ import annotations

import logging
from typing import Any

from breadth.fit import fit_and_record, init_shard

from neighbours import ALLOCATIONS, FIT_SHARD_COUNT, N_DRAWS, allocation_dir
from neighbours.allocation import allocation_frames
from neighbours.census import load_allocations

__all__ = ["decode_shard_index", "run_fit_shard"]

logger = logging.getLogger(__name__)


def decode_shard_index(shard_index: int) -> tuple[int, str]:
    """Decode a shard index into (split_index, allocation) over 3 splits x 2 allocations."""
    if shard_index not in range(FIT_SHARD_COUNT):
        raise ValueError(f"shard_index must be in [0, {FIT_SHARD_COUNT - 1}]")
    names = list(ALLOCATIONS)
    s_idx = shard_index // len(names)
    a_idx = shard_index % len(names)
    return s_idx, names[a_idx]


def run_fit_shard(config: dict[str, Any], shard_index: int) -> None:
    """Execute all draws of one (split, allocation) shard.

    The seed/neighbour/random patients were already drawn by the census
    stage and are read from ``data/allocations.json``, so fitting never
    touches Virchow2 embeddings again; no deep allocation is refit here
    since it coincides with the stored exp-5 G5_m32 grid cell.
    """
    split_idx, allocation = decode_shard_index(shard_index)
    train_df, class_names, evals, paths = init_shard(config, split_idx)
    split_allocations = load_allocations(config)[str(split_idx)]["allocations"]
    g, m = ALLOCATIONS[allocation]
    for draw_idx in range(N_DRAWS):
        logger.info(
            "Fitting split %d, allocation %s, draw %d", split_idx, allocation, draw_idx
        )
        record = split_allocations[str(draw_idx)]
        frames = allocation_frames(train_df, class_names, record)
        result_dir = allocation_dir(paths, allocation, draw_idx)
        fit_and_record(
            config, result_dir, frames[allocation], class_names, (g, m, draw_idx), evals
        )
