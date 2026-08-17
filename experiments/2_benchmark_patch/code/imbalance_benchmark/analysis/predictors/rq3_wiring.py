from __future__ import annotations

import math
from typing import Any

import numpy as np

from imbalance_benchmark.analysis.predictors.hierarchical_models import fit_rq3_model

__all__ = [
    "RQ3Cell",
    "build_predictors",
    "fit_deficit_model",
    "fit_recovery_model",
    "leave_one_group_out",
]


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
    """One dataset-target, severity, and method cell's RQ3 inputs.

    Predictors are achieved ``rho``, ``independent_shortage``,
    ``support_difficulty_alignment``, and ``diversity_shortage``. Outcomes are
    ``deficit_ba`` and gated ``recovery``, each with its bootstrap standard error.
    """


def _cell_predictors(cell: dict[str, Any]) -> list[float]:
    return [
        math.log(cell["rho"]),
        cell["independent_shortage"],
        cell["support_difficulty_alignment"],
        cell["diversity_shortage"],
    ]


def build_predictors(cells: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    """Four-column signal matrix shared by both RQ3 models."""
    predictors = np.array([_cell_predictors(cell) for cell in cells], dtype=np.float64)
    groups = np.array([cell["group"] for cell in cells])
    return predictors, groups


def fit_deficit_model(cells: list[dict[str, Any]]) -> dict[str, Any]:
    """Model signed damage over every cell, including negative deficits.

    Each bootstrap standard error enters the observation variance so a noisy
    deficit estimate is not treated as error-free.
    """
    predictors, groups = build_predictors(cells)
    outcomes = np.array([cell["deficit_ba"] for cell in cells])
    errors = np.array([cell["deficit_se"] for cell in cells])
    return fit_rq3_model(outcomes, predictors, groups, errors, is_logistic=False)


def fit_recovery_model(cells: list[dict[str, Any]]) -> dict[str, Any]:
    """Model recovery only where a prespecified damage gate passed."""
    gated = [cell for cell in cells if cell["gate_passed"]]
    if not gated:
        return {}
    predictors, groups = build_predictors(gated)
    outcomes = np.array([cell["recovery"] for cell in gated])
    errors = np.array([np.nan_to_num(cell["recovery_se"]) for cell in gated])
    return fit_rq3_model(outcomes, predictors, groups, errors, is_logistic=False)


def _held_out_prediction(
    fit: dict[str, Any], mean: np.ndarray, std: np.ndarray, cell: dict[str, Any]
) -> float:
    """Predict one held-out deficit from standardized signal predictors."""
    predictors = np.array(_cell_predictors(cell))
    return float(fit["intercept"] + ((predictors - mean) / std) @ fit["slopes"])


def leave_one_group_out(cells: list[dict[str, Any]]) -> dict[str, Any]:
    """Leave-one-dataset-target-group-out held-out damage prediction."""
    groups = sorted({cell["group"] for cell in cells})
    results: dict[str, Any] = {}
    for held in groups:
        train = [cell for cell in cells if cell["group"] != held]
        test = [cell for cell in cells if cell["group"] == held]
        if len(train) < 2 or not test:
            continue
        fit = fit_deficit_model(train)
        predictors, _ = build_predictors(train)
        mean = predictors.mean(0)
        std = np.maximum(predictors.std(0), 1e-8)
        predictions = [_held_out_prediction(fit, mean, std, cell) for cell in test]
        actual = np.array([cell["deficit_ba"] for cell in test])
        results[held] = {
            "held_out_rmse": float(
                np.sqrt(np.mean((np.array(predictions) - actual) ** 2))
            ),
            "n": len(test),
        }
    return results
