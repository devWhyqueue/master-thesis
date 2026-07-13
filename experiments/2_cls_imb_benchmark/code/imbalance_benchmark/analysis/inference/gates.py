from __future__ import annotations

import math
from typing import Any

import numpy as np

__all__ = [
    "DISCRIMINATION_THRESHOLD",
    "CALIBRATION_THRESHOLD",
    "deficit",
    "recovery",
    "ci_excludes_zero",
    "confidence_interval",
    "discrimination_gate",
    "calibration_gate",
    "gate_cell",
    "discrimination_gate_comparison",
    "calibration_gate_comparison",
]

# Report §"Imbalance deficit, recovery, and inference": the two co-primary gate
# thresholds, evaluated from CE runs only, before any mitigation comparison.
DISCRIMINATION_THRESHOLD = 0.02
CALIBRATION_THRESHOLD = 0.05


def deficit(reference: float, imbalanced: float) -> float:
    """Higher-is-better imbalance-induced deficit: D_M = M(balanced CE) - M(imbalanced CE).

    For the calibration axis the caller passes the sign-corrected pair
    ``NLL_tail(imbalanced CE)`` as ``reference`` and ``NLL_tail(balanced CE)``
    as ``imbalanced`` (lower-is-better, flipped so the result stays a
    higher-is-better deficit), per Eq. recovery's calibration-axis note.
    """
    return reference - imbalanced


def recovery(
    method_metric: float, imbalanced_ce_metric: float, deficit_value: float
) -> float:
    """R_M = (M(method) - M(imbalanced CE)) / D_M (Eq. recovery); NaN if D_M == 0."""
    if deficit_value == 0.0:
        return math.nan
    return (method_metric - imbalanced_ce_metric) / deficit_value


def ci_excludes_zero(ci_low: float, ci_high: float) -> bool:
    """Whether a 95% CI excludes zero (either entirely positive or entirely negative)."""
    return ci_low > 0.0 or ci_high < 0.0


def discrimination_gate(ba_deficit: float, ci: tuple[float, float]) -> bool:
    """Opens when CE's paired BA deficit >= 0.02 and its 95% CI excludes zero."""
    return ba_deficit >= DISCRIMINATION_THRESHOLD and ci_excludes_zero(*ci)


def calibration_gate(tail_nll_deficit: float, ci: tuple[float, float]) -> bool:
    """Opens when CE's tail-group macro-NLL deficit >= 0.05 nats and its 95% CI excludes zero."""
    return tail_nll_deficit >= CALIBRATION_THRESHOLD and ci_excludes_zero(*ci)


def gate_cell(
    ba_deficit: float,
    ba_ci: tuple[float, float],
    cal_deficit: float,
    cal_ci: tuple[float, float],
) -> dict[str, bool]:
    """Evaluate both co-primary gates for one dataset-regime-severity-assignment cell."""
    discrimination = discrimination_gate(ba_deficit, ba_ci)
    calibration = calibration_gate(cal_deficit, cal_ci)
    return {
        "discrimination": discrimination,
        "calibration": calibration,
        "opened": discrimination or calibration,
    }


def confidence_interval(dist: np.ndarray) -> tuple[float, float]:
    """95% percentile confidence interval, ignoring NaNs from a zero-deficit recovery ratio."""
    return (float(np.nanpercentile(dist, 2.5)), float(np.nanpercentile(dist, 97.5)))


def discrimination_gate_comparison(
    severity: str, balanced_ba: np.ndarray, severity_ba: np.ndarray
) -> tuple[dict[str, Any], bool, np.ndarray]:
    """CE-only discrimination-axis deficit and gate check for one severity."""
    deficit_dist = balanced_ba - severity_ba
    ci = confidence_interval(deficit_dist)
    passed = discrimination_gate(float(np.mean(deficit_dist)), ci)
    comparison = {
        "method": "ce",
        "gate": "discrimination",
        "severity": severity,
        "effect": float(np.mean(deficit_dist)),
        "ci": ci,
        "gate_passed": passed,
        "p_value": None,
    }
    return comparison, passed, deficit_dist


def calibration_gate_comparison(
    severity: str,
    balanced_tail_nll: np.ndarray | None,
    severity_tail_nll: np.ndarray | None,
) -> tuple[dict[str, Any], bool, np.ndarray] | None:
    """CE-only calibration-axis (tail macro NLL) deficit and gate check for one severity."""
    if balanced_tail_nll is None or severity_tail_nll is None:
        return None
    deficit_dist = severity_tail_nll - balanced_tail_nll
    ci = confidence_interval(deficit_dist)
    passed = calibration_gate(float(np.mean(deficit_dist)), ci)
    comparison = {
        "method": "ce",
        "gate": "calibration",
        "severity": severity,
        "effect": float(np.mean(deficit_dist)),
        "ci": ci,
        "gate_passed": passed,
        "p_value": None,
    }
    return comparison, passed, deficit_dist
