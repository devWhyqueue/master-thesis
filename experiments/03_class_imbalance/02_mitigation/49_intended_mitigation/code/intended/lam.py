"""Per-lambda logistic-regression path and exp-25's validation selection rule.

``fit_lambda_path`` keeps every lambda's fit (exp-25's ``_select_best_lambda`` discards all but
the winner), so logit adjustment can re-select lambda per tau on the cached path. The selection
replicates exp-25 exactly: lambdas ascending, non-converged fits skipped, a later lambda wins on
a strictly better or tied (``TIE_TOLERANCE``) validation score.
"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any, NamedTuple, cast

import numpy as np
from decodability.linear import LinearFitResult, fit_multinomial_logistic
from imbalance_benchmark.analysis.aggregation.parallel_cache import worker_count
from imbalance_benchmark.analysis.reporting.clustered_endpoints import (
    _macro_recall_mean,
)
from joblib import Parallel, delayed, parallel_config
from scipy.optimize import minimize
from scipy.special import log_softmax

from breadth import LAMBDAS, MAX_ITER, TIE_TOLERANCE, TOLERANCE

__all__ = [
    "fit_lambda_path",
    "GRADIENT_TOL",
    "OffsetFit",
    "fit_offset_logistic",
    "load_path",
    "logits",
    "save_path",
    "select_lambda",
    "validation_scores",
]


def fit_lambda_path(
    x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None
) -> list[LinearFitResult]:
    """One exp-25 logistic fit per ``LAMBDAS`` entry, in grid order, fitted in parallel."""
    with parallel_config(backend="loky", inner_max_num_threads=2):
        return cast(
            list[LinearFitResult],
            Parallel(n_jobs=min(worker_count(), len(LAMBDAS)))(
                delayed(fit_multinomial_logistic)(
                    x,
                    y,
                    lambda_val=lam,
                    tol=TOLERANCE,
                    max_iter=MAX_ITER,
                    sample_weight=sample_weight,
                )
                for lam in LAMBDAS
            ),
        )


def logits(fit: LinearFitResult, x: np.ndarray) -> np.ndarray:
    """Float64 logits, computed exactly as ``predict_logreg`` does before its argmax."""
    return np.asarray(x, dtype=np.float64) @ fit.coef.T + fit.intercept


def validation_scores(
    fits: list[LinearFitResult],
    val_x: np.ndarray,
    val_y: np.ndarray,
    val_cases: np.ndarray,
    offset: np.ndarray | None = None,
) -> list[float | None]:
    """Patient-macro recall mean per fit (``None`` if not converged), ``offset`` added to logits."""
    shift = 0.0 if offset is None else offset
    return [
        _macro_recall_mean(
            val_y, (logits(fit, val_x) + shift).argmax(axis=1), val_cases
        )
        if fit.converged
        else None
        for fit in fits
    ]


def select_lambda(
    fits: list[LinearFitResult],
    val_x: np.ndarray,
    val_y: np.ndarray,
    val_cases: np.ndarray,
    offset: np.ndarray | None = None,
) -> int:
    """Index of the validation-selected lambda; ``offset`` is added to every logit before argmax."""
    best_index, best_score = None, -1.0
    scores = validation_scores(fits, val_x, val_y, val_cases, offset)
    for index, score in enumerate(scores):
        if score is None:
            continue
        diff = score - best_score
        if best_index is None or diff > TIE_TOLERANCE or abs(diff) <= TIE_TOLERANCE:
            best_index, best_score = index, score
    if best_index is None:
        raise RuntimeError("No candidate converged during validation tuning")
    return best_index


class OffsetFit(NamedTuple):
    """Offset fit and its stationarity: converged iff the max gradient entry <= ``GRADIENT_TOL``."""

    coef: np.ndarray
    intercept: np.ndarray
    converged: bool
    gradient_max: float
    iterations: int


GRADIENT_TOL = 1e-8


def _offset_objective(
    theta: np.ndarray, x: np.ndarray, onehot: np.ndarray, offset: np.ndarray, lam: float
) -> tuple[float, np.ndarray]:
    """Penalized loss and its gradient at the flattened ``(W, b)``."""
    (n, dim), k = x.shape, len(offset)
    coef, intercept = theta[: k * dim].reshape(k, dim), theta[k * dim :]
    log_p = log_softmax(x @ coef.T + intercept + offset, axis=1)
    residual = (np.exp(log_p) - onehot) / n
    loss = -float((log_p * onehot).sum()) / n + lam / 2 * float((coef**2).sum())
    grad = np.concatenate([(residual.T @ x + lam * coef).ravel(), residual.sum(0)])
    return loss, grad


def fit_offset_logistic(
    x: np.ndarray, y: np.ndarray, lam: float, offset: np.ndarray
) -> OffsetFit:
    """Train-time logit adjustment: mean CE(x W^T + b + offset) + lam/2 ||W||^2, b unpenalized."""
    x = np.asarray(x, dtype=np.float64)
    dim, k = x.shape[1], len(offset)
    result = minimize(
        _offset_objective,
        np.zeros(k * dim + k),
        args=(x, np.eye(k)[y], offset, lam),
        jac=True,
        method="L-BFGS-B",
        options={"gtol": 1e-10, "ftol": 0.0, "maxiter": MAX_ITER},
    )
    gradient_max = float(np.abs(result.jac).max())
    return OffsetFit(
        result.x[: k * dim].reshape(k, dim),
        result.x[k * dim :],
        gradient_max <= GRADIENT_TOL,
        gradient_max,
        int(result.nit),
    )


def save_path(path: Path, fits: list[LinearFitResult]) -> None:
    """Every lambda's fit, one stacked array per ``LinearFitResult`` field."""
    columns = {
        f.name: np.asarray([getattr(fit, f.name) for fit in fits])
        for f in fields(LinearFitResult)
    }
    np.savez(path, **cast(Any, columns))


def load_path(path: Path) -> list[LinearFitResult]:
    """The cached lambda path written by ``save_path``."""
    with np.load(path) as stored:
        columns = {f.name: stored[f.name] for f in fields(LinearFitResult)}
    return [
        LinearFitResult(
            **{k: v[i] if v.ndim > 1 else v[i].item() for k, v in columns.items()}
        )
        for i in range(len(columns["coef"]))
    ]
