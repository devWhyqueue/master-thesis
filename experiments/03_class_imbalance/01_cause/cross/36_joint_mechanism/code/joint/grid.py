"""Experiment-local lambda grid: wider than ``breadth.LAMBDAS`` (PLAN.md line 54), and every
candidate is retained -- not just the tuned best -- so the fixed-regularization control
(PLAN.md line 39) re-evaluates an already-fit candidate instead of refitting.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
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
from imbalance_benchmark.common import write_json
from joblib import Parallel, delayed, parallel_config

from breadth import MAX_ITER, TIE_TOLERANCE, TOLERANCE
from breadth.fit import EvalPartition

__all__ = [
    "LAMBDAS",
    "GridCandidate",
    "fit_grid",
    "select_best",
    "select_at_lambda",
    "evaluate",
    "write_grid",
    "read_grid",
]

LAMBDAS: tuple[float, ...] = tuple(
    10.0**e for e in range(-8, 3)
)  # 1e-8 .. 1e2, PLAN.md line 54


@dataclass(frozen=True)
class GridCandidate:
    """One lambda's fit, its own validation score, and whether the solver converged."""

    lambda_val: float
    fit: LinearFitResult
    val_score: float


def fit_grid(
    train_x: np.ndarray,
    train_y: np.ndarray,
    evals: EvalPartition,
    sample_weight: np.ndarray | None = None,
) -> list[GridCandidate]:
    """Fit every lambda in ``LAMBDAS``, retaining coefficients and convergence for each."""
    v_cases = evals.val_id["case_id"].astype(str).to_numpy()
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
    candidates = []
    for lam, fit in zip(LAMBDAS, fits):
        score = float("-inf")
        if fit.converged:
            preds, _ = predict_logreg(evals.val_x, fit.coef, fit.intercept)
            score = _macro_recall_mean(evals.val_y, preds, v_cases)
        candidates.append(GridCandidate(lam, fit, score))
    return candidates


def select_best(candidates: list[GridCandidate]) -> GridCandidate:
    """Best validation score, ties broken toward larger lambda (mirrors ``breadth.fit``)."""
    best = candidates[0]
    for cand in candidates[1:]:
        if not cand.fit.converged:
            continue
        diff = cand.val_score - best.val_score
        if not best.fit.converged or diff > TIE_TOLERANCE or abs(diff) <= TIE_TOLERANCE:
            best = cand
    if not best.fit.converged:
        raise RuntimeError("No candidate converged during validation tuning")
    return best


def select_at_lambda(
    candidates: list[GridCandidate], lambda_val: float
) -> GridCandidate:
    """The grid candidate at exactly ``lambda_val`` (the fixed-regularization control)."""
    for cand in candidates:
        if cand.lambda_val == lambda_val:
            return cand
    raise ValueError(f"lambda={lambda_val} is not in this grid")


def evaluate(
    candidate: GridCandidate, evals: EvalPartition
) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    """Validation and test endpoints for one candidate (mirrors ``breadth.fit.tune_and_fit_draw``)."""
    v_cases = evals.val_id["case_id"].astype(str).to_numpy()
    v_slides = evals.val_id["slide_id"].astype(str).to_numpy()
    val_preds, _ = predict_logreg(
        evals.val_x, candidate.fit.coef, candidate.fit.intercept
    )
    val_end = _cluster_discrimination(
        evals.val_y, val_preds, v_cases, v_slides, is_mil=False
    )
    test_preds, test_probs = predict_logreg(
        evals.test_x, candidate.fit.coef, candidate.fit.intercept
    )
    test_end = clustered_endpoints(
        evals.test_y, test_preds, test_probs, evals.test_id, is_mil=False
    )
    return test_preds, test_probs, val_end, test_end


def write_grid(out_dir: Path, candidates: list[GridCandidate]) -> None:
    """Persist every candidate's coefficients (npz) and scalar metadata (json), PLAN.md line 77."""
    out_dir.mkdir(parents=True, exist_ok=True)
    coef = np.stack([c.fit.coef for c in candidates])
    intercept = np.stack([c.fit.intercept for c in candidates])
    np.savez(out_dir / "grid.npz", coef=coef, intercept=intercept)
    meta = [
        {
            "lambda_val": c.lambda_val,
            "c_val": c.fit.c_val,
            "solver": c.fit.solver,
            "precision": c.fit.precision,
            "tolerance": c.fit.tolerance,
            "solver_tolerance": c.fit.solver_tolerance,
            "max_iter": c.fit.max_iter,
            "n_iter": c.fit.n_iter,
            "objective": c.fit.objective,
            "converged": c.fit.converged,
            "weighted": c.fit.weighted,
            "val_score": c.val_score,
        }
        for c in candidates
    ]
    write_json(out_dir / "grid.json", cast(Any, meta))


def _candidate_from_meta(
    m: dict[str, Any], coef: np.ndarray, intercept: np.ndarray
) -> GridCandidate:
    fit = LinearFitResult(
        coef=coef,
        intercept=intercept,
        lambda_val=m["lambda_val"],
        c_val=m["c_val"],
        solver=m["solver"],
        precision=m["precision"],
        tolerance=m["tolerance"],
        solver_tolerance=m["solver_tolerance"],
        max_iter=m["max_iter"],
        n_iter=m["n_iter"],
        objective=m["objective"],
        converged=m["converged"],
        weighted=m["weighted"],
    )
    return GridCandidate(m["lambda_val"], fit, m["val_score"])


def read_grid(out_dir: Path) -> list[GridCandidate]:
    """Reload every candidate written by :func:`write_grid`."""
    meta = json.loads((out_dir / "grid.json").read_text(encoding="utf-8"))
    with np.load(out_dir / "grid.npz") as arrays:
        coef, intercept = arrays["coef"], arrays["intercept"]
    return [_candidate_from_meta(m, coef[i], intercept[i]) for i, m in enumerate(meta)]
