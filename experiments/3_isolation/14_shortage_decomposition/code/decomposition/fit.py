"""Fit stage: one (split, 5-draw block) shard of the shortage-decomposition fits."""

from __future__ import annotations

import logging
from typing import Any, cast

import pandas as pd

from breadth.fit import fit_and_record, init_shard

from sites import allocation_dir

from neighbours.allocation import _patch_rows

from decomposition import N_SPLITS
from decomposition.census import load_allocations

__all__ = ["decode_shard_index", "shard_count", "run_fit_shard"]

logger = logging.getLogger(__name__)

_DRAWS_PER_SHARD = 5


def shard_count(n_draws: int) -> int:
    """Total fit-array shards for ``n_draws`` draws per split: N_SPLITS x n_draws/5."""
    if n_draws % _DRAWS_PER_SHARD != 0:
        raise ValueError(f"n_draws must be a multiple of {_DRAWS_PER_SHARD}")
    return N_SPLITS * (n_draws // _DRAWS_PER_SHARD)


def decode_shard_index(shard_index: int, n_draws: int) -> tuple[int, int]:
    """Decode a shard index into (split_index, draw_block) over N_SPLITS x n_draws/5."""
    blocks = n_draws // _DRAWS_PER_SHARD
    total = N_SPLITS * blocks
    if shard_index not in range(total):
        raise ValueError(f"shard_index must be in [0, {total - 1}]")
    return shard_index // blocks, shard_index % blocks


def _draw_frame(
    train_df: pd.DataFrame, class_names: list[str], draw_rows: dict[str, dict[str, Any]]
) -> pd.DataFrame:
    """Build one draw's training frame from its per-class census allocation."""
    idxs: list[int] = []
    for c_name in class_names:
        row = draw_rows[c_name]
        cls_df = cast(pd.DataFrame, train_df[train_df["cancer_type"] == c_name])
        for patient in row["patients"]:
            idxs.extend(_patch_rows(cls_df, patient, row["m"]))
    return train_df.loc[idxs].copy().reset_index(drop=True)


def run_fit_shard(config: dict[str, Any], shard_index: int, n_draws: int) -> None:
    """Execute one (split, 5-draw block) shard of the shortage-decomposition fits.

    Every class contributes its own (G, m) cohort for the draw, so a fit's
    training frame mixes patient counts across classes; ``fit_and_record``
    does not need to know this, since it only samples a training frame.
    """
    split_idx, block = decode_shard_index(shard_index, n_draws)
    train_df, class_names, evals, paths = init_shard(config, split_idx)
    rows = load_allocations(config, split_idx)["rows"]
    by_draw: dict[int, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_draw.setdefault(row["draw"], {})[row["class"]] = row

    for draw_idx in range(block * _DRAWS_PER_SHARD, (block + 1) * _DRAWS_PER_SHARD):
        logger.info("Fitting split %d, draw %d", split_idx, draw_idx)
        frame = _draw_frame(train_df, class_names, by_draw[draw_idx])
        fit_and_record(
            config,
            allocation_dir(paths, "cells", draw_idx),
            frame,
            class_names,
            (-1, -1, draw_idx),
            evals,
        )
