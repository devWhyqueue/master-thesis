"""Shared shard primitives: the dataset's own G=10 cohort, exp-34's frozen separation artifacts,
one (setting, arm)'s training rows, and the fit-or-load-from-disk step every fit stage shares.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, NamedTuple

import numpy as np
import pandas as pd
from imbalance_benchmark.common import RUN_RECORD_NAME, write_run_record

from breadth.fit import EvalPartition, _build_draw_record

from spectrum import baseline_config

from prevalence.fit import (
    _Shard,
    _arm_rows,
    _prior_weights,
    _shard_context,
    class_counts,
)

from separation.fit import _tcga10_shard
from separation.geometry import Centres, apply_intervention, load_centres
from separation.precheck import load_alpha

from joint import CENTRE_DEPTH, CORRECTION_SETTINGS, FIT_SOURCE, G, SEPARATION_SETTINGS
from joint.grid import (
    GridCandidate,
    evaluate,
    fit_grid,
    read_grid,
    select_best,
    write_grid,
)
from joint.target import move_to_target, move_to_target_negated, shift_target

__all__ = [
    "ShardContext",
    "TrainingData",
    "dataset_shard",
    "separation_artifacts",
    "use_alpha",
    "make_evals_for",
    "training_rows",
    "fit_or_load",
    "build_extra",
]

_ALPHA_KEY: dict[str, str] = {"bracs": "alpha_expand", "tcga_ut": "alpha_contract"}


@dataclass(frozen=True)
class ShardContext:
    """Everything every setting/arm/control of one (split, draw) shard shares."""

    config: dict[str, Any]
    paths: dict[str, Path]
    shard: _Shard
    centres: Centres
    alpha: float
    dense_target_native: np.ndarray
    names: list[str]
    draw_idx: int


class TrainingData(NamedTuple):
    """One (setting, arm)'s training features, targets, allocation, and prior weight."""

    x: np.ndarray
    y: np.ndarray
    counts: list[int]
    prior_counts: list[int] | None
    weight: np.ndarray | None


def dataset_shard(
    dataset: str,
    train_df: pd.DataFrame,
    names: list[str],
    split_idx: int,
    draw_idx: int,
) -> _Shard:
    """This dataset's own G=10 cohort: TCGA-UT nests exp-34's own draw, BRACS is native G=10."""
    if dataset == "tcga_ut":
        return _tcga10_shard(train_df, names, split_idx, draw_idx)
    return _shard_context(train_df, names, split_idx, draw_idx, G)


def separation_artifacts(
    config: dict[str, Any], dataset: str, split_idx: int, names: list[str]
) -> tuple[Centres, float]:
    """Exp-34's own frozen native centres and per-split alpha for this dataset, read-only."""
    sep_config = baseline_config(config, "separation_outputs")
    precheck = load_alpha(sep_config)
    alpha = float(precheck[_ALPHA_KEY[dataset]][split_idx])
    centres = load_centres(sep_config, split_idx, names)
    return centres, alpha


def use_alpha(setting: str, alpha: float) -> float:
    """This setting's own alpha: the separation settings use it, the others stay at 1.0."""
    return alpha if setting in SEPARATION_SETTINGS else 1.0


def _eval_partition(
    evals: EvalPartition, centres: Centres, alpha: float
) -> EvalPartition:
    """Evaluation features receive only the separation transformation, never centre correction."""
    if alpha == 1.0:
        return evals
    return replace(
        evals,
        val_x=apply_intervention(evals.val_x, evals.val_y, centres, alpha),
        test_x=apply_intervention(evals.test_x, evals.test_y, centres, alpha),
    )


def make_evals_for(
    evals: EvalPartition, centres: Centres, alpha: float
) -> Callable[[str], EvalPartition]:
    """A memoized per-setting eval partition: native/centre_only share alpha=1, the rest share alpha."""
    cache: dict[float, EvalPartition] = {}

    def _for(setting: str) -> EvalPartition:
        alpha_here = use_alpha(setting, alpha)
        if alpha_here not in cache:
            cache[alpha_here] = _eval_partition(evals, centres, alpha_here)
        return cache[alpha_here]

    return _for


def training_rows(
    ctx: ShardContext, setting: str, arm: str, negate: bool
) -> TrainingData:
    """One (setting, arm)'s training features, targets, allocation, and prior weight."""
    shard = ctx.shard
    data_arm, prior_arm = FIT_SOURCE[arm]
    counts = class_counts(
        data_arm, shard.perm, shard.available, shard.pool_counts, shard.g
    )
    x, y = _arm_rows(shard.train_df, shard.names, shard.patients, counts)
    prior_counts = None
    weight = None
    if prior_arm is not None:
        prior_counts = class_counts(
            prior_arm, shard.perm, shard.available, shard.pool_counts, shard.g
        )
        weight = _prior_weights(counts, prior_counts)
    alpha_here = use_alpha(setting, ctx.alpha)
    if alpha_here != 1.0:
        x = apply_intervention(x, y, ctx.centres, alpha_here)
    if setting in CORRECTION_SETTINGS:
        target = shift_target(ctx.dense_target_native, ctx.centres, alpha_here)
        x = (
            move_to_target_negated(x, y, target)
            if negate
            else move_to_target(x, y, target)
        )
    return TrainingData(x, y, counts, prior_counts, weight)


def fit_or_load(
    config: dict[str, Any],
    out_dir: Path,
    data: TrainingData,
    evals: EvalPartition,
    draw_idx: int,
    extra: dict[str, Any],
) -> list[GridCandidate]:
    """Load an already-completed fit's grid, or fit and persist it (resumable, signed shards)."""
    if (out_dir / RUN_RECORD_NAME).exists() and (out_dir / "grid.json").exists():
        return read_grid(out_dir)
    candidates = fit_grid(data.x, data.y, evals, data.weight)
    best = select_best(candidates)
    eval_outs = evaluate(best, evals)
    rec = _build_draw_record(
        config,
        (G, CENTRE_DEPTH, draw_idx),
        best.lambda_val,
        best.fit,
        eval_outs,
        evals.test_y,
    )
    write_run_record(out_dir, {**rec, **extra}, keep_arrays=True)
    write_grid(out_dir, candidates)
    return candidates


def build_extra(
    setting: str,
    arm: str,
    alpha: float,
    names: list[str],
    data: TrainingData,
    wrong_direction: bool = False,
) -> dict[str, Any]:
    """The JSON-serializable extras stored alongside one (setting, arm)'s run record."""
    extra: dict[str, Any] = {
        "setting": setting,
        "arm": arm,
        "alpha": alpha,
        "wrong_direction": wrong_direction,
        "class_counts": dict(zip(names, (int(c) for c in data.counts))),
    }
    if data.prior_counts is not None:
        extra["prior_counts"] = dict(zip(names, (int(c) for c in data.prior_counts)))
    return extra
