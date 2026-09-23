"""Fit stage: every setting/arm/control of one (split, draw) shard.

One shard is one (split, draw): the same drawn cohort (``joint.fitting.shard.dataset_shard``) and
the same dense same-cohort target (``joint.target.dense_cohort_target``) feed all four settings'
B/P/S/R, so "exact cohort/allocation pairing" (PLAN.md gate 1) holds by construction, not by a
later check. Reuses exp-25/27/28's allocator and prior reweighting (``prevalence.fit``) and exp-34's
frozen alpha and native centres (``separation.geometry``/``separation.precheck``) unchanged; only
this experiment's own lambda grid and centre-correction target are new.
"""

from __future__ import annotations

from typing import Any, Callable

from breadth.fit import EvalPartition, init_shard

from sites import allocation_dir

from joint import ARMS, CENTRE_DEPTH, MAIN_DRAWS, N_SPLITS, PILOT_DRAWS, SETTINGS
from joint.fitting.controls import fit_fixed_lambda, fit_wrong_direction
from joint.fitting.shard import (
    ShardContext,
    build_extra,
    dataset_shard,
    fit_or_load,
    make_evals_for,
    separation_artifacts,
    training_rows,
    use_alpha,
)
from joint.grid import GridCandidate
from joint.target import dense_cohort_target

__all__ = [
    "pilot_shard_count",
    "main_shard_count",
    "decode_shard_index",
    "run_fit_shard",
]


def pilot_shard_count() -> int:
    """Pilot-array shards: one per (split, draw), draws 0-1 (PLAN.md "Execution and precheck")."""
    return N_SPLITS * len(PILOT_DRAWS)


def main_shard_count() -> int:
    """Main-array shards: one per (split, draw), locked draws 2-9."""
    return N_SPLITS * len(MAIN_DRAWS)


def _decode(draws: tuple[int, ...], shard_index: int) -> tuple[int, int]:
    count = N_SPLITS * len(draws)
    if shard_index not in range(count):
        raise ValueError(f"shard_index must be in [0, {count - 1}]")
    split_idx, draw_pos = divmod(shard_index, len(draws))
    return split_idx, draws[draw_pos]


def decode_shard_index(phase: str, shard_index: int) -> tuple[int, int]:
    """Decode a shard index into (split_idx, draw_idx)."""
    return _decode(PILOT_DRAWS if phase == "pilot" else MAIN_DRAWS, shard_index)


def _fit_core_settings(
    ctx: ShardContext, evals_for: Callable[[str], EvalPartition]
) -> dict[str, list[GridCandidate]]:
    """Tuned B/P/S/R of every setting (the 16-evaluation core of PLAN.md's 25-per-shard total)."""
    candidates_by_key: dict[str, list[GridCandidate]] = {}
    for setting in SETTINGS:
        setting_evals = evals_for(setting)
        for arm in ARMS:
            data = training_rows(ctx, setting, arm, negate=False)
            extra = build_extra(
                setting, arm, use_alpha(setting, ctx.alpha), ctx.names, data
            )
            candidates_by_key[f"{setting}_{arm}"] = fit_or_load(
                ctx.config,
                allocation_dir(ctx.paths, f"{setting}_{arm}", ctx.draw_idx),
                data,
                setting_evals,
                ctx.draw_idx,
                extra,
            )
    return candidates_by_key


def _build_context(
    config: dict[str, Any], phase: str, shard_index: int
) -> tuple[ShardContext, EvalPartition]:
    dataset = config["dataset"]["name"]
    split_idx, draw_idx = decode_shard_index(phase, shard_index)
    train_df, names, evals, paths = init_shard(config, split_idx)
    shard = dataset_shard(dataset, train_df, names, split_idx, draw_idx)
    centres, alpha = separation_artifacts(config, dataset, split_idx, names)
    dense_target_native = dense_cohort_target(
        train_df, names, shard.patients, CENTRE_DEPTH
    )
    ctx = ShardContext(
        config, paths, shard, centres, alpha, dense_target_native, names, draw_idx
    )
    return ctx, evals


def run_fit_shard(config: dict[str, Any], phase: str, shard_index: int) -> None:
    """Fit every setting/arm/control of one (split, draw) shard."""
    ctx, evals = _build_context(config, phase, shard_index)
    evals_for = make_evals_for(evals, ctx.centres, ctx.alpha)
    candidates_by_key = _fit_core_settings(ctx, evals_for)
    fit_wrong_direction(ctx, evals_for("joint"))
    fit_fixed_lambda(ctx, candidates_by_key, evals_for)
