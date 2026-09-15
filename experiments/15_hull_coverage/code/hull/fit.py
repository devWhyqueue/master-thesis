"""Fit stage: for every draw, one classifier per patient count (G=5, G=10, G=20), all classes at that count."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from breadth.fit import fit_and_record, init_shard

from sites import allocation_dir

from decomposition.census import load_allocations
from decomposition.fit import _draw_frame

from hull import N_SPLITS
from hull.design import CELLS

__all__ = ["shard_count", "decode_shard_index", "fit_dir", "run_fit_shard"]

logger = logging.getLogger(__name__)

_DRAWS_PER_SHARD = 5


def shard_count(n_draws: int) -> int:
    """Fit-array shards for ``n_draws`` draws per split: one per split and five-draw block."""
    if n_draws % _DRAWS_PER_SHARD != 0:
        raise ValueError(f"n_draws must be a multiple of {_DRAWS_PER_SHARD}")
    return N_SPLITS * (n_draws // _DRAWS_PER_SHARD)


def decode_shard_index(shard_index: int, n_draws: int) -> tuple[int, int]:
    """Decode a shard index into (split_index, draw_block)."""
    blocks = n_draws // _DRAWS_PER_SHARD
    if shard_index not in range(shard_count(n_draws)):
        raise ValueError(f"shard_index must be in [0, {shard_count(n_draws) - 1}]")
    return shard_index // blocks, shard_index % blocks


def fit_dir(paths: dict[str, Path], g: int, draw_idx: int) -> Path:
    """Result directory of the classifier trained with ``g`` patients per class in one draw."""
    return allocation_dir(paths, f"g{g}", draw_idx)


def run_fit_shard(config: dict[str, Any], shard_index: int, n_draws: int) -> None:
    """Fit the three uniform-patient-count classifiers of every draw in one shard."""
    split_idx, block = decode_shard_index(shard_index, n_draws)
    train_df, class_names, evals, paths = init_shard(config, split_idx)
    by_draw_g: dict[tuple[int, int], dict[str, dict[str, Any]]] = {}
    for row in load_allocations(config, split_idx)["rows"]:
        by_draw_g.setdefault((row["draw"], row["g"]), {})[row["class"]] = row

    for draw_idx in range(block * _DRAWS_PER_SHARD, (block + 1) * _DRAWS_PER_SHARD):
        for g in CELLS:
            logger.info("Fitting split %d, draw %d, G=%d", split_idx, draw_idx, g)
            frame = _draw_frame(train_df, class_names, by_draw_g[(draw_idx, g)])
            meta = (g, CELLS[g][0].m, draw_idx)
            fit_and_record(
                config, fit_dir(paths, g, draw_idx), frame, class_names, meta, evals
            )
