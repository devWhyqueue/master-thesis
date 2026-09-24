"""Fit stage (phase 2 only): native B/P/S/R of one (split, draw) shard, draws 2-9 -- 4 grids/shard,
not exp-36's 25. Alpha and centre correction are unused at the native setting (PLAN.md fit.py
reuse list), so the shard context carries placeholder values instead of loading exp-34's frozen
separation artifacts.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np

from breadth.fit import init_shard

from sites import allocation_dir

from joint.fitting.shard import (
    ShardContext,
    build_extra,
    dataset_shard,
    fit_or_load,
    training_rows,
)

from selection_rule import ARMS, MAIN_DRAWS, N_SPLITS

__all__ = ["shard_count", "decode_shard_index", "run_fit_shard"]


def shard_count() -> int:
    """One shard per (split, draw), locked draws 2-9."""
    return N_SPLITS * len(MAIN_DRAWS)


def decode_shard_index(shard_index: int) -> tuple[int, int]:
    """Decode a shard index into (split_idx, draw_idx); split varies slowest, draw fastest."""
    count = shard_count()
    if shard_index not in range(count):
        raise ValueError(f"shard_index must be in [0, {count - 1}]")
    split_idx, draw_pos = divmod(shard_index, len(MAIN_DRAWS))
    return split_idx, MAIN_DRAWS[draw_pos]


def _fit_arm(
    ctx: ShardContext, evals: Any, paths: dict[str, Any], names: list[str], arm: str
) -> None:
    data = training_rows(ctx, "native", arm, negate=False)
    extra = build_extra("native", arm, 1.0, names, data)
    fit_or_load(
        ctx.config,
        allocation_dir(paths, f"native_{arm}", ctx.draw_idx),
        data,
        evals,
        ctx.draw_idx,
        extra,
    )


def run_fit_shard(config: dict[str, Any], shard_index: int) -> None:
    """Tuned native B/P/S/R of one (split, draw) shard."""
    dataset = config["dataset"]["name"]
    split_idx, draw_idx = decode_shard_index(shard_index)
    train_df, names, evals, paths = init_shard(config, split_idx)
    shard = dataset_shard(dataset, train_df, names, split_idx, draw_idx)
    ctx = ShardContext(
        config,
        paths,
        shard,
        cast(Any, None),
        1.0,
        cast(np.ndarray, None),
        names,
        draw_idx,
    )
    for arm in ARMS:
        _fit_arm(ctx, evals, paths, names, arm)
