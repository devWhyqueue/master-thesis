"""Fit stage: S/St arms of one (split, draw) shard, cohort-only James-Stein centre shrinkage.

St records fitted on a smaller alpha grid are extended in place: only the missing factors are
fit, and the record is rewritten with the new fit only if one of them wins validation selection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from breadth import TIE_TOLERANCE
from breadth.fit import EvalPartition, _build_draw_record, init_shard, tune_and_fit_draw
from imbalance_benchmark.common import (
    RUN_RECORD_NAME,
    read_run_record,
    write_json,
    write_run_record,
)

from sites import allocation_dir

from centre import PATIENT_COUNTS, patches_per_patient
from centre.arms import move_centres
from centre.cohort import TrainingTable, draw_cohorts, training_table
from centre.fit import decode_shard_index, shard_count, split_arm

from shrinkage import ALPHA_FACTORS, ARMS
from shrinkage.centres import patient_means, shrunk_centres

__all__ = ["decode_shard_index", "shard_count", "select_alpha", "run_fit_shard"]


def select_alpha(scores: dict[str, float]) -> str:
    """Factor key ``_tune_alpha`` would pick from these validation scores (ties to larger alpha)."""
    best, best_score = "", -1.0
    for key in sorted(scores, key=float):
        if not best or scores[key] >= best_score - TIE_TOLERANCE:
            best, best_score = key, scores[key]
    return best


def _done_scores(out_dir: Path) -> dict[str, float]:
    """Stored alpha validation scores of an arm's run record, empty if not fitted yet."""
    record = read_run_record(out_dir, array_fields=())
    return {} if record is None else record["alpha_validation_scores"]


def _tune_alpha(
    means: Any,
    x: Any,
    y: Any,
    evals: EvalPartition,
    factors: tuple[float, ...],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Tune alpha on validation, moving only the training centres; ties go to the larger alpha."""
    best: tuple[Any, ...] | None = None
    best_score, best_alpha, scores = -1.0, 0.0, {}
    for factor in factors:
        moved = move_centres(x, y, shrunk_centres(means, factor))
        out = tune_and_fit_draw(moved, y, evals)
        score = float(out[4]["patient_macro_balanced_accuracy"])
        scores[str(factor)] = score
        if best is None or score >= best_score - TIE_TOLERANCE:
            best, best_score, best_alpha = out, score, factor
    assert best is not None
    return best, {"alpha_factor": best_alpha, "alpha_validation_scores": scores}


def _fit_s(
    config: dict[str, Any],
    out_dir: Path,
    arm: str,
    table: TrainingTable,
    means: Any,
    evals: EvalPartition,
    draw_idx: int,
    g: int,
) -> None:
    """Fit the closed-form S arm (alpha = 1, no tuning)."""
    moved = move_centres(table.x, table.y, shrunk_centres(means, 1.0))
    fit, lam, test_preds, test_probs, val_end, test_end = tune_and_fit_draw(
        moved, table.y, evals
    )
    meta = (g, patches_per_patient(g), draw_idx)
    rec = _build_draw_record(
        config,
        meta,
        lam,
        fit,
        (test_preds, test_probs, val_end, test_end),
        evals.test_y,
    )
    write_run_record(out_dir, {**rec, "arm": arm}, keep_arrays=True)


def _fit_st(
    config: dict[str, Any],
    out_dir: Path,
    arm: str,
    table: TrainingTable,
    means: Any,
    evals: EvalPartition,
    draw_idx: int,
    g: int,
) -> None:
    """Fit the tuned St arm, extending a partial alpha grid in place."""
    old = read_run_record(out_dir, array_fields=())
    done = _done_scores(out_dir)
    missing = tuple(f for f in ALPHA_FACTORS if str(f) not in done)
    out, extra = _tune_alpha(means, table.x, table.y, evals, missing)
    scores = {**done, **extra["alpha_validation_scores"]}
    if old is not None and select_alpha(scores) in done:
        write_json(
            out_dir / RUN_RECORD_NAME, {**old, "alpha_validation_scores": scores}
        )
        return
    fit, lam, test_preds, test_probs, val_end, test_end = out
    meta = (g, patches_per_patient(g), draw_idx)
    rec = _build_draw_record(
        config,
        meta,
        lam,
        fit,
        (test_preds, test_probs, val_end, test_end),
        evals.test_y,
    )
    write_run_record(
        out_dir,
        {
            **rec,
            "arm": arm,
            "alpha_factor": extra["alpha_factor"],
            "alpha_validation_scores": scores,
        },
        keep_arrays=True,
    )


def _st_pending(out_dir: Path) -> bool:
    """Whether an St arm still has alpha factors missing from its run record."""
    return bool(set(map(str, ALPHA_FACTORS)) - set(_done_scores(out_dir)))


def _arm_pending(paths: dict[str, Path], arm: str, draw_idx: int) -> bool:
    """Whether an arm still needs fitting: a missing record, or (for St) a missing alpha factor."""
    out_dir = allocation_dir(paths, arm, draw_idx)
    if split_arm(arm)[0] == "St":
        return _st_pending(out_dir)
    return not (out_dir / RUN_RECORD_NAME).exists()


def run_fit_shard(config: dict[str, Any], shard_index: int) -> None:
    """Fit every pending S/St arm of one (split, draw) shard, extending partial St grids."""
    split_idx, draw_idx = decode_shard_index(shard_index)
    train_df, names, evals, paths = init_shard(config, split_idx)
    pending = [a for a in ARMS if _arm_pending(paths, a, draw_idx)]
    if not pending:
        return
    cohorts = draw_cohorts(train_df, names, split_idx, draw_idx)
    tables = {
        g: training_table(train_df, names, cohorts.nested, g) for g in PATIENT_COUNTS
    }
    for arm in pending:
        family, g = split_arm(arm)
        out_dir = allocation_dir(paths, arm, draw_idx)
        means = patient_means(tables[g], len(names), g)
        if family == "S":
            _fit_s(config, out_dir, arm, tables[g], means, evals, draw_idx, g)
        else:
            _fit_st(config, out_dir, arm, tables[g], means, evals, draw_idx, g)
