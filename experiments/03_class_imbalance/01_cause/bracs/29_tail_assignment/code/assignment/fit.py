"""Fit stage: exp-29's new shifted-tail arms of one (split, draw) shard.

r1/r{rho} themselves are reused from exp-26's stored outputs (shift k = 0), not fit here.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from imbalance_benchmark.common import RUN_RECORD_NAME

from breadth.fit import init_shard

from centre.fit import decode_shard_index, shard_count

from sites import allocation_dir

from prevalence import patients_per_class
from prevalence.fit import _fit_arm, _shard_context

from assignment import NEW_FIT_ARMS, RATIOS, SHIFTS

__all__ = ["decode_shard_index", "shard_count", "shifted_perm", "run_fit_shard"]


def shifted_perm(perm: np.ndarray, k: int) -> np.ndarray:
    """Cyclic rotation of a class-to-rank permutation by ``k`` ranks (k = 0 is ``perm`` itself)."""
    return np.roll(perm, k)


def _arm_name(k: int, rho: int) -> str:
    return f"a{k}_r{rho}"


def _fit_shift(
    config: dict[str, Any],
    paths: dict[str, Any],
    shard: Any,
    evals: Any,
    draw_idx: int,
    pending: list[str],
    rho: int,
    k: int,
) -> None:
    """Fit one pending (shift, rho) arm, or skip it if already fit."""
    arm = _arm_name(k, rho)
    if arm not in pending:
        return
    shifted_shard = shard._replace(perm=shifted_perm(shard.perm, k))
    _fit_arm(
        config,
        allocation_dir(paths, arm, draw_idx),
        arm,
        shifted_shard,
        evals,
        draw_idx,
        data_arm=f"r{rho}",
    )


def run_fit_shard(config: dict[str, Any], shard_index: int) -> None:
    """Fit every pending shifted-tail arm of one (split, draw)."""
    split_idx, draw_idx = decode_shard_index(shard_index)
    train_df, names, evals, paths = init_shard(config, split_idx)
    pending = [
        a
        for a in NEW_FIT_ARMS
        if not (allocation_dir(paths, a, draw_idx) / RUN_RECORD_NAME).exists()
    ]
    if not pending:
        return
    shard = _shard_context(
        train_df, names, split_idx, draw_idx, patients_per_class(config)
    )
    for rho in RATIOS:
        for k in SHIFTS:
            _fit_shift(config, paths, shard, evals, draw_idx, pending, rho, k)
