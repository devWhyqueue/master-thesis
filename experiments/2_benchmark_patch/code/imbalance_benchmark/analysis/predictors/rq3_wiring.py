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
    "fit_linked_sensitivity_models",
    "leave_one_group_out",
]

# Descriptive covariates entered only as single-predictor sensitivity fits, never
# jointly with the two primary predictors (report §"Predictors of damage...").
SENSITIVITY_COVARIATES = (
    "separability",
    "learnability",
    "log_min_support",
    "log_min_patient_support",
    "log_effective_support",
    "is_wsi",
)


def support_difficulty_alignment(
    allocated: dict[str, Any], freeze: dict[str, Any]
) -> float:
    """Correlate log allocated support with frozen class difficulty."""
    difficulty = freeze.get("difficulty_evidence", {}).get("difficulty", {})
    names = list(allocated.get("allocated_counts", {}))
    if not names or any(name not in difficulty for name in names):
        raise RuntimeError("Frozen difficulty evidence is required for RQ3 alignment")
    support = np.log([allocated["allocated_counts"][name] for name in names])
    scores = np.asarray([difficulty[name] for name in names], dtype=float)
    return (
        0.0
        if np.ptp(support) == 0 or np.ptp(scores) == 0
        else float(np.corrcoef(support, scores)[0, 1])
    )


class RQ3Cell(dict):
    """One dataset-target x severity x method cell's RQ3 inputs.

    Required keys: ``group`` (dataset-target group id, gets a random
    intercept), ``rho`` (achieved imbalance ratio),
    ``support_difficulty_alignment`` (support/difficulty correlation),
    ``gate_passed`` (bool), ``deficit_ba``
    (float), ``deficit_se`` (bootstrap standard error of ``deficit_ba``),
    ``recovery`` (float, meaningful only when ``gate_passed``), and
    ``recovery_se``.
    """


def build_predictors(cells: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    """Standardizable primary matrix: [log achieved rho, support/difficulty alignment]."""
    x = np.array(
        [[math.log(c["rho"]), c["support_difficulty_alignment"]] for c in cells],
        dtype=np.float64,
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
    out: dict[str, Any] = {}
    for name in SENSITIVITY_COVARIATES:
        eligible = [cell for cell in cells if name in cell]
        if not eligible:
            continue
        x = np.array([[float(cell[name])] for cell in eligible], dtype=np.float64)
        groups = np.array([cell["group"] for cell in eligible])
        y = np.array([cell["deficit_ba"] for cell in eligible])
        s_errors = np.array([cell["deficit_se"] for cell in eligible])
        out[name] = fit_rq3_model(y, x, groups, s_errors, is_logistic=False)
    return out


def fit_linked_sensitivity_models(
    deficit_cells: list[dict[str, Any]], recovery_cells: list[dict[str, Any]]
) -> dict[str, Any]:
    """Fit every descriptive sensitivity predictor for all three linked outcomes."""
    out: dict[str, Any] = {}
    for name in SENSITIVITY_COVARIATES:
        gate_pass = _sensitivity_fit(deficit_cells, name, "gate_pass")
        deficit = _sensitivity_fit(deficit_cells, name, "deficit")
        if not gate_pass and not deficit:
            continue
        out[name] = {
            "gate_pass": gate_pass,
            "deficit": deficit,
            "recovery": _sensitivity_fit(recovery_cells, name, "recovery"),
        }
    return out


def _sensitivity_fit(
    cells: list[dict[str, Any]], name: str, outcome: str
) -> dict[str, Any]:
    """Fit one sensitivity covariate for one linked RQ3 outcome."""
    selected = [
        cell
        for cell in cells
        if name in cell and (outcome != "recovery" or cell["gate_passed"])
    ]
    if not selected:
        return {}
    x = np.array([[float(cell[name])] for cell in selected], dtype=np.float64)
    groups = np.array([cell["group"] for cell in selected])
    if outcome == "gate_pass":
        y = np.array([float(cell["gate_passed"]) for cell in selected])
        errors, logistic = np.zeros(len(selected)), True
    else:
        value = "deficit_ba" if outcome == "deficit" else "recovery"
        error = "deficit_se" if outcome == "deficit" else "recovery_se"
        y = np.array([cell[value] for cell in selected])
        errors = np.array([np.nan_to_num(cell[error]) for cell in selected])
        logistic = False
    return fit_rq3_model(y, x, groups, errors, is_logistic=logistic)


def _held_out_prediction(
    fit: dict[str, Any], mean: np.ndarray, std: np.ndarray, cell: dict[str, Any]
) -> float:
    """Predict one held-out deficit from standardized primary predictors."""
    predictors = np.array([math.log(cell["rho"]), cell["support_difficulty_alignment"]])
    return float(fit["intercept"] + ((predictors - mean) / std) @ fit["slopes"])


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
        preds = [_held_out_prediction(fit, mean, std, cell) for cell in test]
        actual = np.array([c["deficit_ba"] for c in test])
        out[held] = {
            "held_out_rmse": float(np.sqrt(np.mean((np.array(preds) - actual) ** 2))),
            "n": len(test),
        }
    return out
