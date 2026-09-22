"""Fit stage: exp-30's easy/hard-tail r100 arms of one (split, draw) shard.

r1/r10/r100/N themselves are reused unchanged from exp-25's stored outputs, not refit here.
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

from permutation import NEW_FIT_ARMS, RATIO_B
from permutation.order import order_perm, tail_order

__all__ = ["decode_shard_index", "shard_count", "run_fit_shard"]


def _fit_direction(
    config: dict[str, Any],
    paths: dict[str, Any],
    shard: Any,
    evals: Any,
    draw_idx: int,
    pending: list[str],
    order: list[str],
    names: list[str],
    easy: bool,
) -> None:
    """Fit one pending (easy/hard) direction, or skip it if already fit."""
    arm = f"{'easy' if easy else 'hard'}_r{RATIO_B}"
    if arm not in pending:
        return
    tail_shard = shard._replace(perm=order_perm(order, names, easy))
    _fit_arm(
        config,
        allocation_dir(paths, arm, draw_idx),
        arm,
        tail_shard,
        evals,
        draw_idx,
        data_arm=f"r{RATIO_B}",
    )


def run_fit_shard(config: dict[str, Any], shard_index: int) -> None:
    """Fit every pending easy/hard-tail arm of one (split, draw)."""
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
    order = tail_order(exp25_config, canonical_class_names(config))
    shard = _shard_context(
        train_df, names, split_idx, draw_idx, patients_per_class(config)
    )
    for easy in (True, False):
        _fit_direction(
            config, paths, shard, evals, draw_idx, pending, order, names, easy
        )
