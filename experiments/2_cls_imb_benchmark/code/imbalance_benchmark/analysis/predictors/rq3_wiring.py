from __future__ import annotations

import math
from typing import Any

import numpy as np

from imbalance_benchmark.analysis.predictors.hierarchical_models import fit_rq3_model

__all__ = [
    "RQ3Cell",
    "build_predictors",
    "fit_gate_pass_model",
    "fit_deficit_model",
    "fit_recovery_model",
    "fit_sensitivity_models",
    "leave_one_group_out",
]

# Descriptive covariates entered only as single-predictor sensitivity fits, never
# jointly with the two primary predictors (report §"Predictors of damage...").
SENSITIVITY_COVARIATES = ("learnability", "log_min_support", "is_wsi")


class RQ3Cell(dict):
    """One dataset-target x severity x method cell's RQ3 inputs.

    Required keys: ``group`` (dataset-target group id, gets a random
    intercept), ``rho`` (requested imbalance ratio), ``separability``
    (intrinsic separability score), ``gate_passed`` (bool), ``deficit_ba``
    (float), ``deficit_se`` (bootstrap standard error of ``deficit_ba``),
    ``recovery`` (float, meaningful only when ``gate_passed``), and
    ``recovery_se``.
    """


def build_predictors(cells: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    """Standardizable two-predictor design matrix: [log(rho), intrinsic separability]."""
    x = np.array(
        [[math.log(c["rho"]), c["separability"]] for c in cells], dtype=np.float64
    )
    groups = np.array([c["group"] for c in cells])
    return x, groups


def fit_gate_pass_model(cells: list[dict[str, Any]]) -> dict[str, Any]:
    """RQ3 model 1: logistic model of whether a cell passes either deficit gate."""
    x, groups = build_predictors(cells)
    y = np.array([1.0 if c["gate_passed"] else 0.0 for c in cells])
    s_errors = np.zeros(len(cells))
    return fit_rq3_model(y, x, groups, s_errors, is_logistic=True)


def fit_deficit_model(cells: list[dict[str, Any]]) -> dict[str, Any]:
    """RQ3 model 2: continuous model of D_BA over every cell, including negative deficits.

    Incorporates each cell's bootstrap standard error so the observation
    model uses variance ``s_j^2 + sigma^2`` rather than treating the
    deficit estimate as error-free.
    """
    x, groups = build_predictors(cells)
    y = np.array([c["deficit_ba"] for c in cells])
    s_errors = np.array([c["deficit_se"] for c in cells])
    return fit_rq3_model(y, x, groups, s_errors, is_logistic=False)


def fit_recovery_model(cells: list[dict[str, Any]]) -> dict[str, Any]:
    """RQ3 model 3: recovery, restricted to gated cells only, conditional on demonstrated damage."""
    gated = [c for c in cells if c["gate_passed"]]
    if not gated:
        return {}
    x, groups = build_predictors(gated)
    y = np.array([c["recovery"] for c in gated])
    s_errors = np.array([np.nan_to_num(c["recovery_se"]) for c in gated])
    return fit_rq3_model(y, x, groups, s_errors, is_logistic=False)


def fit_sensitivity_models(cells: list[dict[str, Any]]) -> dict[str, Any]:
    """One single-predictor deficit fit per descriptive covariate (never entered jointly)."""
    groups = np.array([c["group"] for c in cells])
    y = np.array([c["deficit_ba"] for c in cells])
    s_errors = np.array([c["deficit_se"] for c in cells])
    out: dict[str, Any] = {}
    for name in SENSITIVITY_COVARIATES:
        if not all(name in c for c in cells):
            continue
        x = np.array([[float(c[name])] for c in cells], dtype=np.float64)
        out[name] = fit_rq3_model(y, x, groups, s_errors, is_logistic=False)
    return out


def leave_one_group_out(cells: list[dict[str, Any]]) -> dict[str, Any]:
    """Leave-one-dataset-target-group-out held-out deficit prediction."""
    groups = sorted({c["group"] for c in cells})
    out: dict[str, Any] = {}
    for held in groups:
        train = [c for c in cells if c["group"] != held]
        test = [c for c in cells if c["group"] == held]
        if len(train) < 2 or not test:
            continue
        fit = fit_deficit_model(train)
        x, _ = build_predictors(train)
        mean, std = x.mean(0), np.maximum(x.std(0), 1e-8)
        sl = np.array(fit["slopes"], dtype=float)
        preds = [
            fit["intercept"]
            + float(
                ((np.array([np.log(c["rho"]), c["separability"]]) - mean) / std) @ sl
            )
            for c in test
        ]
        actual = np.array([c["deficit_ba"] for c in test])
        out[held] = {
            "held_out_rmse": float(np.sqrt(np.mean((np.array(preds) - actual) ** 2))),
            "n": len(test),
        }
    return out
