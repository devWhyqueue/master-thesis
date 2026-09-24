"""Per-arm tuning, fitting, candidate storage, and temperature calibration.

Every candidate lambda's coefficients and validation score are stored (not just the
selected candidate's), so a later B-selected-lambda sensitivity can be generated from
stored coefficients without refitting (plans/04_implementation.md).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple, cast

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config

from decodability.linear import (
    LinearFitResult,
    fit_multinomial_logistic,
    predict_logreg,
)
from imbalance_benchmark.analysis.aggregation.parallel_cache import worker_count
from imbalance_benchmark.analysis.reporting.clustered_endpoints import (
    _cluster_discrimination,
    _macro_recall_mean,
    clustered_endpoints,
)
from imbalance_benchmark.common import write_json, write_run_record
from imbalance_benchmark.manifest.statistics import achieved_rho

from breadth.calibrate import (
    TEMPERATURE_NAME,
    _calibration_record,
    _logits,
    _store_if_inexact,
)
from breadth.fit import EvalPartition, _build_draw_record

from prevalence import DEPTH
from prevalence.fit import _arm_rows, _prior_weights, _Shard, class_counts

from transfer import LAMBDAS, MAX_ITER, TIE_TOLERANCE, TOLERANCE

__all__ = ["CANDIDATES_NAME", "tune_and_fit_draw", "fit_arm"]

CANDIDATES_NAME = "candidates.npz"


def _select_best_lambda(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    val_id: pd.DataFrame,
    sample_weight: np.ndarray | None = None,
) -> tuple[LinearFitResult, float, dict[str, Any], list[dict[str, Any]]]:
    """Grid-search exp-39's frozen lambda grid on validation; ties go to the larger lambda.

    Returns the selected fit/lambda/validation-endpoints, plus every candidate's
    (converged fit, validation score) for later provenance and sensitivity use.
    """
    v_cases = val_id["case_id"].astype(str).to_numpy()
    v_slides = val_id["slide_id"].astype(str).to_numpy()
    with parallel_config(backend="loky", inner_max_num_threads=2):
        fits = cast(
            list[LinearFitResult],
            Parallel(n_jobs=min(worker_count(), len(LAMBDAS)))(
                delayed(fit_multinomial_logistic)(
                    train_x,
                    train_y,
                    lambda_val=lam,
                    tol=TOLERANCE,
                    max_iter=MAX_ITER,
                    sample_weight=sample_weight,
                )
                for lam in LAMBDAS
            ),
        )

    best_score, best_lam = -1.0, LAMBDAS[0]
    best_fit: LinearFitResult | None = None
    best_preds: np.ndarray | None = None
    candidates: list[dict[str, Any]] = []

    for lam, fit in zip(LAMBDAS, fits):
        score = None
        if fit.converged:
            preds, _ = predict_logreg(val_x, fit.coef, fit.intercept)
            score = _macro_recall_mean(val_y, preds, v_cases)
            diff = score - best_score
            if best_fit is None or diff > TIE_TOLERANCE or abs(diff) <= TIE_TOLERANCE:
                best_score, best_lam, best_fit, best_preds = score, lam, fit, preds
        candidates.append(
            {
                "lambda": lam,
                "fit": fit,
                "converged": fit.converged,
                "validation_score": score,
            }
        )

    if best_fit is None or best_preds is None:
        raise RuntimeError("No candidate converged during validation tuning")
    best_end = _cluster_discrimination(
        val_y, best_preds, v_cases, v_slides, is_mil=False
    )
    return best_fit, best_lam, best_end, candidates


def tune_and_fit_draw(
    train_x: np.ndarray,
    train_y: np.ndarray,
    evals: EvalPartition,
    sample_weight: np.ndarray | None = None,
) -> tuple[
    LinearFitResult,
    float,
    np.ndarray,
    np.ndarray,
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
]:
    """Select the best candidate on exp-39's frozen grid and evaluate it on test."""
    best_fit, best_lam, val_end, candidates = _select_best_lambda(
        train_x, train_y, evals.val_x, evals.val_y, evals.val_id, sample_weight
    )
    test_preds, test_probs = predict_logreg(
        evals.test_x, best_fit.coef, best_fit.intercept
    )
    test_end = clustered_endpoints(
        evals.test_y, test_preds, test_probs, evals.test_id, is_mil=False
    )
    return best_fit, best_lam, test_preds, test_probs, val_end, test_end, candidates


def _save_candidates(out_dir: Path, candidates: list[dict[str, Any]]) -> None:
    """Every candidate's coefficients/intercept, indexed to ``transfer.LAMBDAS`` order.

    Kept out of the JSON run record (``write_run_record`` would inline them) since
    even one arm's 11 candidates of (n_classes, feature_dim) coefficients are too
    large for readable JSON; a compressed sidecar bounds this to coefficients and
    validation scores only, never a candidate's full test probabilities.
    """
    coef = np.stack([c["fit"].coef for c in candidates])
    intercept = np.stack([c["fit"].intercept for c in candidates])
    np.savez_compressed(
        out_dir / CANDIDATES_NAME,
        lambdas=np.asarray(LAMBDAS, dtype=np.float64),
        coef=coef,
        intercept=intercept,
    )


def _write_temperature(
    out_dir: Path,
    evals: EvalPartition,
    fit: LinearFitResult,
    test_preds: np.ndarray,
    test_probs: np.ndarray,
    lam: float,
) -> None:
    """Fit a validation temperature from the in-memory fit and store it alongside the run record."""
    val_logits = _logits(evals.val_x, fit.coef, fit.intercept)
    test_logits = _logits(evals.test_x, fit.coef, fit.intercept)
    payload, scaled = _calibration_record(evals, val_logits, test_logits, test_preds)
    payload["max_abs_scaled_probability_error"] = _store_if_inexact(
        out_dir, scaled, test_probs, payload["temperature"]
    )
    write_json(out_dir / TEMPERATURE_NAME, {**payload, "selected_lambda": lam})


def _candidate_summary(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = ("lambda", "converged", "validation_score")
    return [{k: c[k] for k in keys} for c in candidates]


class _Allocation(NamedTuple):
    """One arm's realized training rows, targets, and (optional) prior reweighting."""

    x: np.ndarray
    y: np.ndarray
    counts: list[int]
    prior_counts: list[int] | None
    weight: np.ndarray | None


def _allocate(shard: _Shard, fit_source: tuple[str, str | None]) -> _Allocation:
    """Allocate one arm's patch counts and, when it has a prior arm, its row weights."""
    data_arm, prior_arm = fit_source
    counts = class_counts(
        data_arm, shard.perm, shard.available, shard.pool_counts, shard.g
    )
    x, y = _arm_rows(shard.train_df, shard.names, shard.patients, counts)
    if prior_arm is None:
        return _Allocation(x, y, counts, None, None)
    prior_counts = class_counts(
        prior_arm, shard.perm, shard.available, shard.pool_counts, shard.g
    )
    weight = _prior_weights(counts, prior_counts)
    return _Allocation(x, y, counts, prior_counts, weight)


def _evidence_fields(
    arm: str, shard: _Shard, allocation: _Allocation, candidates: list[dict[str, Any]]
) -> dict[str, Any]:
    """Non-solver run-record fields: arm identity, realized allocation, and candidate summary."""
    named_counts = dict(zip(shard.names, (int(c) for c in allocation.counts)))
    extra: dict[str, Any] = {
        "arm": arm,
        "class_counts": named_counts,
        "realized_rho": achieved_rho(named_counts),
        "candidates": _candidate_summary(candidates),
    }
    if allocation.prior_counts is not None:
        extra["prior_counts"] = dict(
            zip(shard.names, (int(c) for c in allocation.prior_counts))
        )
    return extra


def fit_arm(
    config: dict[str, Any],
    out_dir: Path,
    arm: str,
    shard: _Shard,
    evals: EvalPartition,
    draw_idx: int,
    fit_source: tuple[str, str | None],
) -> None:
    """Allocate one arm's patch counts, fit it on the frozen grid, and write its evidence."""
    allocation = _allocate(shard, fit_source)
    fit, lam, test_preds, test_probs, val_end, test_end, candidates = tune_and_fit_draw(
        allocation.x, allocation.y, evals, allocation.weight
    )
    rec = _build_draw_record(
        config,
        (shard.g, DEPTH, draw_idx),
        lam,
        fit,
        (test_preds, test_probs, val_end, test_end),
        evals.test_y,
    )
    extra = _evidence_fields(arm, shard, allocation, candidates)
    write_run_record(out_dir, {**rec, **extra}, keep_arrays=True)
    _save_candidates(out_dir, candidates)
    _write_temperature(out_dir, evals, fit, test_preds, test_probs, lam)
