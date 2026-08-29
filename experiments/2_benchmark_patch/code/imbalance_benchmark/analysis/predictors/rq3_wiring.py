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


def _alignment_identifiable(cell: dict[str, Any]) -> bool:
    """Whether this cell's support-difficulty-alignment predictor is identifiable.

    Pearson correlation over exactly two points (a binary target) is always
    +/-1, saturating the alignment predictor and confounding its pooled
    coefficient. Cells missing ``n_classes`` (hand-built cells predating this
    field, e.g. in tests) are treated as identifiable rather than excluded.
    """
    n_classes = cell.get("n_classes")
    return n_classes is None or n_classes > 2


def fit_deficit_model(cells: list[dict[str, Any]]) -> dict[str, Any]:
    """Model signed damage over every cell, including negative deficits.

    Each bootstrap standard error enters the observation variance so a noisy
    deficit estimate is not treated as error-free. Cells from a binary-target
    dataset are excluded: the shared design matrix has one alignment column,
    and a saturated predictor for those cells would confound its coefficient
    for every dataset.
    """
    identifiable = [cell for cell in cells if _alignment_identifiable(cell)]
    if not identifiable:
        return {}
    predictors, groups = build_predictors(identifiable)
    outcomes = np.array([cell["deficit_ba"] for cell in identifiable])
    errors = np.array([cell["deficit_se"] for cell in identifiable])
    return fit_rq3_model(outcomes, predictors, groups, errors, is_logistic=False)


def _has_defined_recovery(cell: dict[str, Any]) -> bool:
    """Whether this cell's recovery fraction exists as a finite number.

    The matched-versus-unmatched contrast rows carry no deficit denominator, so
    their recovery is undefined; leaving them in propagates NaN through the fit.
    """
    recovery = cell.get("recovery")
    return recovery is not None and math.isfinite(recovery)


def fit_recovery_model(cells: list[dict[str, Any]]) -> dict[str, Any]:
    """Model recovery only where a prespecified damage gate passed.

    Binary-target cells are excluded for the same reason as
    :func:`fit_deficit_model`: their alignment predictor is saturated.
    """
    gated = [
        cell
        for cell in cells
        if cell["gate_passed"]
        and _has_defined_recovery(cell)
        and _alignment_identifiable(cell)
    ]
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
