"""L2-regularized multinomial logistic regression probe."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

__all__ = ["LinearFitResult", "fit_multinomial_logistic", "predict_logreg"]


@dataclass(frozen=True)
class LinearFitResult:
    """Fitted logistic model and solver metadata."""

    coef: np.ndarray
    intercept: np.ndarray
    lambda_val: float
    c_val: float
    solver: str
    precision: str
    tolerance: float
    solver_tolerance: float
    max_iter: int
    n_iter: int
    objective: float
    converged: bool
    weighted: bool


def _compute_objective(
    features: np.ndarray,
    labels: np.ndarray,
    coef: np.ndarray,
    intercept: np.ndarray,
    lambda_val: float,
    sample_weight: np.ndarray | None = None,
) -> float:
    """Compute mean cross-entropy loss + lambda / 2 * ||W||_F^2."""
    logits = features @ coef.T + intercept
    max_logits = np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    n_samples = len(labels)
    chosen_probs = np.clip(probs[np.arange(n_samples), labels], 1e-15, 1.0)
    log_probs = np.log(chosen_probs)
    mean_ce = -float(
        np.mean(log_probs)
        if sample_weight is None
        else np.average(log_probs, weights=sample_weight)
    )
    l2_penalty = (lambda_val / 2.0) * float(np.sum(coef**2))
    return mean_ce + l2_penalty


def _fit_once(
    features: np.ndarray,
    labels: np.ndarray,
    c_val: float,
    max_iter: int,
    tol: float,
    sample_weight: np.ndarray | None = None,
) -> tuple[LogisticRegression, bool]:
    """Run one L-BFGS fit and track ConvergenceWarning."""
    model = LogisticRegression(
        penalty="l2",
        solver="lbfgs",
        fit_intercept=True,
        tol=tol,
        max_iter=max_iter,
        C=c_val,
        multi_class="multinomial",
        random_state=0,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(features, labels, sample_weight=sample_weight)
        has_conv_warn = any(issubclass(w.category, ConvergenceWarning) for w in caught)
    return model, not has_conv_warn


def _build_result(
    model: LogisticRegression,
    feat64: np.ndarray,
    lab_int: np.ndarray,
    lambda_val: float,
    c_val: float,
    tol: float,
    solver_tol: float,
    max_iter: int,
    conv: bool,
    weight64: np.ndarray | None,
) -> LinearFitResult:
    """Assemble a LinearFitResult from a fitted model and its solver settings."""
    w = np.asarray(model.coef_, dtype=np.float64)
    b = np.asarray(model.intercept_, dtype=np.float64)
    obj = _compute_objective(feat64, lab_int, w, b, lambda_val, weight64)
    return LinearFitResult(
        w,
        b,
        lambda_val,
        c_val,
        "lbfgs",
        "float64",
        tol,
        solver_tol,
        max_iter,
        int(model.n_iter_[0]),
        obj,
        conv,
        weight64 is not None,
    )


def fit_multinomial_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    lambda_val: float,
    tol: float = 1e-8,
    max_iter: int = 10000,
    sample_weight: np.ndarray | None = None,
) -> LinearFitResult:
    """Fit with C = 1 / (N * lambda) and ``tol`` as a gradient tolerance on the mean
    objective; scikit-learn applies its own ``tol`` to the summed objective, so the
    solver receives ``tol * N``. ``sample_weight``, when given, must sum to N so this
    mapping stays valid.
    """
    feat64, lab_int = np.asarray(features, np.float64), np.asarray(labels, np.int64)
    n_samples = feat64.shape[0]
    weight64 = None if sample_weight is None else np.asarray(sample_weight, np.float64)
    c_val, solver_tol = 1.0 / (n_samples * lambda_val), tol * n_samples
    model, conv = _fit_once(feat64, lab_int, c_val, max_iter, solver_tol, weight64)
    return _build_result(
        model,
        feat64,
        lab_int,
        lambda_val,
        c_val,
        tol,
        solver_tol,
        max_iter,
        conv,
        weight64,
    )


def predict_logreg(
    features: np.ndarray, coef: np.ndarray, intercept: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Predict probabilities and classes via fixed-order argmax."""
    logits = np.asarray(features, dtype=np.float64) @ coef.T + intercept
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    return np.argmax(logits, axis=1), probs
