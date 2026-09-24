"""Fit stage: one (split, draw) shard's B arm (tuned) and three S arms (each tuned, full grid
stored), then the three S arms again at B's own tuned lambda -- the fixed-lambda control the
primary analysis actually reads (PLAN.md "Arms per (split, draw)").
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from breadth.fit import EvalPartition, init_shard

from sites import allocation_dir

from joint.fitting.controls import _write_fixed
from joint.fitting.shard import TrainingData, dataset_shard, fit_or_load
from joint.grid import GridCandidate, select_at_lambda, select_best

from support import ARMS, MAIN_DRAWS, N_SPLITS, S_ARMS
from support.shard import ClassPool, arm_training_data, b_training_data, class_pools

__all__ = ["shard_count", "decode_shard_index", "run_fit_shard"]


def shard_count() -> int:
    """One shard per (split, draw), locked draws 2-9 (PLAN.md "Regime")."""
    return N_SPLITS * len(MAIN_DRAWS)


def decode_shard_index(shard_index: int) -> tuple[int, int]:
    """Decode a shard index into (split_idx, draw_idx); split varies slowest, draw fastest."""
    count = shard_count()
    if shard_index not in range(count):
        raise ValueError(f"shard_index must be in [0, {count - 1}]")
    split_idx, draw_pos = divmod(shard_index, len(MAIN_DRAWS))
    return split_idx, MAIN_DRAWS[draw_pos]


def _extra(arm: str, names: list[str], data: TrainingData) -> dict[str, Any]:
    return {"arm": arm, "class_counts": dict(zip(names, (int(c) for c in data.counts)))}


def _fit_core_arms(
    config: dict[str, Any],
    paths: dict[str, Path],
    pools: list[ClassPool],
    evals: EvalPartition,
    draw_idx: int,
    names: list[str],
) -> dict[str, list[GridCandidate]]:
    """Tuned B and every S arm of one shard, full grid stored for each."""
    candidates_by_arm: dict[str, list[GridCandidate]] = {}
    for arm in ARMS:
        data = b_training_data(pools) if arm == "B" else arm_training_data(pools, arm)
        candidates_by_arm[arm] = fit_or_load(
            config,
            allocation_dir(paths, arm, draw_idx),
            data,
            evals,
            draw_idx,
            _extra(arm, names, data),
        )
    return candidates_by_arm


def _fit_fixed_lambda_arms(
    config: dict[str, Any],
    paths: dict[str, Path],
    candidates_by_arm: dict[str, list[GridCandidate]],
    evals: EvalPartition,
    draw_idx: int,
) -> None:
    """Every S arm re-evaluated at the B arm's own tuned lambda; no new optimization."""
    b_lambda = select_best(candidates_by_arm["B"]).lambda_val
    for arm in S_ARMS:
        candidate = select_at_lambda(candidates_by_arm[arm], b_lambda)
        _write_fixed(
            config,
            allocation_dir(paths, f"{arm}_fixedlambda", draw_idx),
            candidate,
            evals,
            draw_idx,
            {"arm": arm, "fixed_lambda": True, "source_lambda": b_lambda},
        )


def run_fit_shard(config: dict[str, Any], shard_index: int) -> None:
    """Fit B and every S arm of one (split, draw) shard, then S arms again at B's fixed lambda."""
    dataset = config["dataset"]["name"]
    split_idx, draw_idx = decode_shard_index(shard_index)
    train_df, names, evals, paths = init_shard(config, split_idx)
    shard = dataset_shard(dataset, train_df, names, split_idx, draw_idx)
    pools = class_pools(shard)
    candidates_by_arm = _fit_core_arms(config, paths, pools, evals, draw_idx, names)
    _fit_fixed_lambda_arms(config, paths, candidates_by_arm, evals, draw_idx)
