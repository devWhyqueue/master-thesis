"""One (split, draw) shard's four rules (tuned/oracle/naive/fixed), for every native arm.

Source is exp-36's own ``native_{arm}`` dirs (phase 1, pilot draws) or this experiment's own
(phase 2, main draws, written by ``selection_rule.fit``); either way every rule re-predicts an
already-fit grid, no new optimization.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from imbalance_benchmark.common import RUN_RECORD_NAME

from breadth.fit import EvalPartition, init_shard

from sites import allocation_dir

from joint.fitting.controls import _write_fixed
from joint.grid import read_grid, select_at_lambda, select_best

from selection_rule import ARMS, ORACLE_FOLD_SEED
from selection_rule.fit import decode_shard_index
from selection_rule.oracle import naive_oracle, write_oracle_record

__all__ = ["run_rules", "run_rules_shard"]


def _copy_allocation(src: Path, dst: Path) -> None:
    if (dst / RUN_RECORD_NAME).exists():
        return
    shutil.copytree(src, dst, dirs_exist_ok=True)


def _run_arm_rules(
    config: dict[str, Any],
    dest_paths: dict[str, Path],
    source_paths: dict[str, Path],
    draw_idx: int,
    evals: EvalPartition,
    arm: str,
    b_lambda: float,
) -> None:
    """Tuned/oracle/naive/fixed allocations for one native arm, one (split, draw) shard."""
    src_dir = allocation_dir(source_paths, f"native_{arm}", draw_idx)
    candidates = read_grid(src_dir)
    _copy_allocation(src_dir, allocation_dir(dest_paths, f"tuned_{arm}", draw_idx))
    write_oracle_record(
        config,
        allocation_dir(dest_paths, f"oracle_{arm}", draw_idx),
        candidates,
        evals,
        draw_idx,
        ORACLE_FOLD_SEED,
        {"arm": arm},
    )
    _write_fixed(
        config,
        allocation_dir(dest_paths, f"naive_{arm}", draw_idx),
        naive_oracle(candidates, evals),
        evals,
        draw_idx,
        {"arm": arm, "naive_oracle": True},
    )
    _write_fixed(
        config,
        allocation_dir(dest_paths, f"fixed_{arm}", draw_idx),
        select_at_lambda(candidates, b_lambda),
        evals,
        draw_idx,
        {"arm": arm, "fixed_lambda": True, "source_lambda": b_lambda},
    )


def run_rules(
    config: dict[str, Any],
    dest_paths: dict[str, Path],
    source_paths: dict[str, Path],
    draw_idx: int,
    evals: EvalPartition,
) -> None:
    """Tuned/oracle/naive/fixed allocations for every native arm, one (split, draw) shard."""
    b_dir = allocation_dir(source_paths, "native_B", draw_idx)
    b_lambda = select_best(read_grid(b_dir)).lambda_val
    for arm in ARMS:
        _run_arm_rules(config, dest_paths, source_paths, draw_idx, evals, arm, b_lambda)


def run_rules_shard(config: dict[str, Any], shard_index: int) -> None:
    """Phase 2: one (split, draw) shard's rules, source and dest both this run's own native fits."""
    split_idx, draw_idx = decode_shard_index(shard_index)
    _, _, evals, paths = init_shard(config, split_idx)
    run_rules(config, paths, paths, draw_idx, evals)
