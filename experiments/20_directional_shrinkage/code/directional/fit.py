"""Fit stage: A/At arms of one (split, draw) shard, cohort-only directional centre shrinkage.

At records fitted on a smaller alpha grid are extended in place: only missing nonzero factors
are fit, rewriting the record only if one of them wins. Alpha = 0 is never fit: its score and,
if it wins, its record are read straight from R's arm of ``slurm.baseline_outputs``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from imbalance_benchmark.common import (
    RUN_RECORD_NAME,
    read_run_record,
    write_json,
    write_run_record,
)

from breadth import TIE_TOLERANCE
from breadth.fit import EvalPartition, _build_draw_record, init_shard, tune_and_fit_draw

from sites import allocation_dir

from centre import PATIENT_COUNTS, patches_per_patient
from centre.arms import move_centres
from centre.cohort import TrainingTable, draw_cohorts, training_table
from centre.fit import decode_shard_index, shard_count, split_arm

from directions.basis import cohort_eigenbasis

from shrinkage.centres import patient_means

from directional import (
    ARMS,
    NONZERO_ALPHA_FACTORS,
    baseline_arm_dir,
    baseline_arm_score,
)
from directional.centres import shrunk_centres

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
    basis: Any,
    eigvals: Any,
    g: int,
    x: Any,
    y: Any,
    evals: EvalPartition,
    factors: tuple[float, ...],
    r_score: float,
    prefilled: dict[float, tuple[tuple[Any, ...], float]] | None = None,
) -> tuple[tuple[Any, ...] | None, dict[str, Any]]:
    """Tune alpha on validation; alpha = 0 uses ``r_score`` instead of a fresh fit.

    ``prefilled`` supplies factors already fit elsewhere this shard (the A arm's alpha=1.0
    fit), so the identical matrix is not refit. Returns ``None`` in place of a fit when alpha
    = 0 wins, signalling the caller to reuse the baseline R arm's record wholesale.
    """
    prefilled = prefilled or {}
    scores: dict[str, float] = {"0.0": r_score}
    fits: dict[str, tuple[Any, ...] | None] = {"0.0": None}
    for factor in factors:
        if factor in prefilled:
            out, score = prefilled[factor]
        else:
            moved = move_centres(x, y, shrunk_centres(means, basis, eigvals, g, factor))
            out = tune_and_fit_draw(moved, y, evals)
            score = float(out[4]["patient_macro_balanced_accuracy"])
        fits[str(factor)], scores[str(factor)] = out, score
    best_key = select_alpha(scores)
    return fits[best_key], {
        "alpha_factor": float(best_key),
        "alpha_validation_scores": scores,
    }


def _write_fit_record(
    config: dict[str, Any],
    out_dir: Path,
    arm: str,
    g: int,
    draw_idx: int,
    out: tuple[Any, ...],
    evals: EvalPartition,
    extra: dict[str, Any] | None = None,
) -> None:
    """Assemble and write one draw's run record from a ``tune_and_fit_draw`` result."""
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
    write_run_record(out_dir, {**rec, "arm": arm, **(extra or {})}, keep_arrays=True)


def _fit_a(
    config: dict[str, Any],
    out_dir: Path,
    arm: str,
    table: TrainingTable,
    means: Any,
    basis: Any,
    eigvals: Any,
    evals: EvalPartition,
    draw_idx: int,
    g: int,
) -> tuple[tuple[Any, ...], float]:
    """Fit the fixed A arm (alpha = 1, no tuning); return the fit and its validation score."""
    moved = move_centres(
        table.x, table.y, shrunk_centres(means, basis, eigvals, g, 1.0)
    )
    out = tune_and_fit_draw(moved, table.y, evals)
    _write_fit_record(config, out_dir, arm, g, draw_idx, out, evals)
    return out, float(out[4]["patient_macro_balanced_accuracy"])


def _fit_at(
    config: dict[str, Any],
    out_dir: Path,
    arm: str,
    table: TrainingTable,
    means: Any,
    basis: Any,
    eigvals: Any,
    evals: EvalPartition,
    draw_idx: int,
    g: int,
    split_idx: int,
    prefilled: dict[float, tuple[tuple[Any, ...], float]] | None = None,
) -> None:
    """Fit the tuned At arm, extending a partial alpha grid in place."""
    old = read_run_record(out_dir, array_fields=())
    done = _done_scores(out_dir)
    missing = tuple(f for f in NONZERO_ALPHA_FACTORS if str(f) not in done)
    r_score = baseline_arm_score(config, split_idx, draw_idx, f"R{g}")
    out, extra = _tune_alpha(
        means, basis, eigvals, g, table.x, table.y, evals, missing, r_score, prefilled
    )
    scores = {**done, **extra["alpha_validation_scores"]}
    if old is not None and select_alpha(scores) in done:
        write_json(
            out_dir / RUN_RECORD_NAME, {**old, "alpha_validation_scores": scores}
        )
        return
    if out is None:
        # alpha = 0 wins: reuse the baseline R arm's record wholesale, since directional
        # shrinkage at alpha = 0 leaves this shard's features identical to R's.
        record = read_run_record(baseline_arm_dir(config, split_idx, draw_idx, f"R{g}"))
        assert record is not None
        write_run_record(
            out_dir,
            {
                **record,
                "arm": arm,
                "alpha_factor": 0.0,
                "alpha_validation_scores": scores,
            },
            keep_arrays=True,
        )
        return
    _write_fit_record(
        config,
        out_dir,
        arm,
        g,
        draw_idx,
        out,
        evals,
        {"alpha_factor": extra["alpha_factor"], "alpha_validation_scores": scores},
    )


def _at_pending(out_dir: Path) -> bool:
    """Whether an At arm still has a nonzero alpha factor missing from its run record."""
    return bool(set(map(str, NONZERO_ALPHA_FACTORS)) - set(_done_scores(out_dir)))


def _arm_pending(paths: dict[str, Path], arm: str, draw_idx: int) -> bool:
    """Whether an arm still needs fitting: a missing record, or (for At) a missing alpha factor."""
    out_dir = allocation_dir(paths, arm, draw_idx)
    if split_arm(arm)[0] == "At":
        return _at_pending(out_dir)
    return not (out_dir / RUN_RECORD_NAME).exists()


def _fit_pending_arm(
    config: dict[str, Any],
    paths: dict[str, Path],
    tables: dict[int, TrainingTable],
    names: list[str],
    evals: EvalPartition,
    split_idx: int,
    draw_idx: int,
    arm: str,
    a_cache: dict[int, tuple[tuple[Any, ...], float]],
) -> None:
    """Fit one pending arm; A{g}'s fit is cached so a same-shard At{g} reuses its alpha=1.0 fit."""
    family, g = split_arm(arm)
    out_dir = allocation_dir(paths, arm, draw_idx)
    table = tables[g]
    means = patient_means(table, len(names), g)
    basis, eigvals = cohort_eigenbasis(table, len(names), g)
    common = (config, out_dir, arm, table, means, basis, eigvals, evals, draw_idx, g)
    if family == "A":
        a_cache[g] = _fit_a(*common)
    else:
        prefilled = {1.0: a_cache[g]} if g in a_cache else None
        _fit_at(*common, split_idx, prefilled)


def run_fit_shard(config: dict[str, Any], shard_index: int) -> None:
    """Fit every pending A/At arm of one (split, draw) shard, extending partial At grids."""
    split_idx, draw_idx = decode_shard_index(shard_index)
    train_df, names, evals, paths = init_shard(config, split_idx)
    pending = [a for a in ARMS if _arm_pending(paths, a, draw_idx)]
    if not pending:
        return
    cohorts = draw_cohorts(train_df, names, split_idx, draw_idx)
    tables = {
        g: training_table(train_df, names, cohorts.nested, g) for g in PATIENT_COUNTS
    }
    a_cache: dict[int, tuple[tuple[Any, ...], float]] = {}
    for arm in pending:
        _fit_pending_arm(
            config, paths, tables, names, evals, split_idx, draw_idx, arm, a_cache
        )
