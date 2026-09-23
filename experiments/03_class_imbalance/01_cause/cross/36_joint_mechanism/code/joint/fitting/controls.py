"""Wrong-direction and fixed-regularization controls (PLAN.md lines 39-40): both reuse the same
shard's cohort and, for fixed lambda, an already-fit grid candidate -- no new optimization.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from imbalance_benchmark.common import RUN_RECORD_NAME, write_run_record

from breadth.fit import EvalPartition, _build_draw_record

from sites import allocation_dir

from joint import CENTRE_DEPTH, FIXED_LAMBDA_ARMS_BY_SETTING, G, WRONG_DIRECTION_ARMS
from joint.grid import GridCandidate, evaluate, select_at_lambda, select_best
from joint.fitting.shard import (
    ShardContext,
    build_extra,
    fit_or_load,
    training_rows,
    use_alpha,
)

__all__ = ["fit_wrong_direction", "fit_fixed_lambda"]


def fit_wrong_direction(ctx: ShardContext, joint_evals: EvalPartition) -> None:
    """Joint S/R with the correction vector negated: same magnitude, opposite direction."""
    for arm in WRONG_DIRECTION_ARMS:
        key = f"joint_{arm}_wrong"
        data = training_rows(ctx, "joint", arm, negate=True)
        extra = build_extra(
            "joint",
            arm,
            use_alpha("joint", ctx.alpha),
            ctx.names,
            data,
            wrong_direction=True,
        )
        fit_or_load(
            ctx.config,
            allocation_dir(ctx.paths, key, ctx.draw_idx),
            data,
            joint_evals,
            ctx.draw_idx,
            extra,
        )


def _write_fixed(
    config: dict[str, Any],
    out_dir: Path,
    candidate: GridCandidate,
    evals: EvalPartition,
    draw_idx: int,
    extra: dict[str, Any],
) -> None:
    """Re-evaluate an already-fit grid candidate at the fixed lambda; no new optimization."""
    if (out_dir / RUN_RECORD_NAME).exists():
        return
    eval_outs = evaluate(candidate, evals)
    rec = _build_draw_record(
        config,
        (G, CENTRE_DEPTH, draw_idx),
        candidate.lambda_val,
        candidate.fit,
        eval_outs,
        evals.test_y,
    )
    write_run_record(out_dir, {**rec, **extra}, keep_arrays=True)


def fit_fixed_lambda(
    ctx: ShardContext,
    candidates_by_key: dict[str, list[GridCandidate]],
    evals_for: Callable[[str], EvalPartition],
) -> None:
    """Native's own P/S/R and joint's B/P/S/R, re-evaluated at native-B's tuned lambda."""
    native_b_lambda = select_best(candidates_by_key["native_B"]).lambda_val
    for setting, arms in FIXED_LAMBDA_ARMS_BY_SETTING.items():
        setting_evals = evals_for(setting)
        for arm in arms:
            src_key = f"{setting}_{arm}"
            candidate = select_at_lambda(candidates_by_key[src_key], native_b_lambda)
            extra = {
                "setting": setting,
                "arm": arm,
                "fixed_lambda": True,
                "source_lambda": native_b_lambda,
            }
            _write_fixed(
                ctx.config,
                allocation_dir(ctx.paths, f"{src_key}_fixedlambda", ctx.draw_idx),
                candidate,
                setting_evals,
                ctx.draw_idx,
                extra,
            )
