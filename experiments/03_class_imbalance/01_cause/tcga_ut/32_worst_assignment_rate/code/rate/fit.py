"""Fit stage: exp-32's worst/flip/mild r100 arms of one (split, draw) shard.

r1/r10/r100/N are reused unchanged from exp-25's stored outputs; easy_r100/hard_r100 are reused
unchanged from exp-30's stored outputs. Only the three rate-score-derived arms are fit here.
"""

from __future__ import annotations

from typing import Any

from imbalance_benchmark.common import RUN_RECORD_NAME

from breadth.analyze.canonical import canonical_class_names
from breadth.fit import init_shard

from centre.fit import decode_shard_index, shard_count

from sites import allocation_dir

from spectrum import baseline_config

from prevalence import patients_per_class
from prevalence.fit import _fit_arm, _shard_context

from worst.order import order_perm

from rate import ARM_ORDER, NEW_FIT_ARMS, RATIO_SEARCH
from rate.order import derive_orders

__all__ = ["decode_shard_index", "shard_count", "run_fit_shard"]


def _fit_new_arm(
    config: dict[str, Any],
    paths: dict[str, Any],
    shard: Any,
    evals: Any,
    draw_idx: int,
    pending: list[str],
    orders: dict[str, list[str]],
    names: list[str],
    arm: str,
) -> None:
    """Fit one pending new-order arm, or skip it if already fit."""
    if arm not in pending:
        return
    arm_shard = shard._replace(perm=order_perm(orders[ARM_ORDER[arm]], names))
    _fit_arm(
        config,
        allocation_dir(paths, arm, draw_idx),
        arm,
        arm_shard,
        evals,
        draw_idx,
        data_arm=f"r{RATIO_SEARCH}",
    )


def run_fit_shard(config: dict[str, Any], shard_index: int) -> None:
    """Fit every pending worst/flip/mild arm of one (split, draw)."""
    split_idx, draw_idx = decode_shard_index(shard_index)
    train_df, names, evals, paths = init_shard(config, split_idx)
    pending = [
        a
        for a in NEW_FIT_ARMS
        if not (allocation_dir(paths, a, draw_idx) / RUN_RECORD_NAME).exists()
    ]
    if not pending:
        return
    exp25_config = baseline_config(config, "prevalence_outputs")
    g = patients_per_class(exp25_config)
    orders = derive_orders(exp25_config, canonical_class_names(config), g)
    shard = _shard_context(train_df, names, split_idx, draw_idx, g)
    for arm in NEW_FIT_ARMS:
        _fit_new_arm(config, paths, shard, evals, draw_idx, pending, orders, names, arm)
