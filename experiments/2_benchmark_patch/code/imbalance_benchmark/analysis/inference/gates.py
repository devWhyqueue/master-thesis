from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from imbalance_benchmark.analysis.inference.context import BootstrapContext
from imbalance_benchmark.analysis.inference.confirmatory.holm import PRIMARY_METHODS
from imbalance_benchmark.analysis.inference.permutation import (
    paired_block_permutation_ba,
    paired_block_permutation_tail_nll,
)

__all__ = [
    "DISCRIMINATION_THRESHOLD",
    "CALIBRATION_THRESHOLD",
    "deficit",
    "recovery",
    "ci_excludes_zero",
    "confidence_interval",
    "discrimination_gate",
    "calibration_gate",
    "discrimination_gate_comparison",
    "calibration_gate_comparison",
    "_SeverityInputs",
    "_recovery_comparison",
    "_method_discrimination_recovery",
    "_method_calibration_recovery",
]

# Report §"Imbalance deficit, recovery, and inference": the two co-primary gate
# thresholds. Derived per PLAN_3 §2 (see derive_deficit_thresholds.py); both
# noise-floor terms bound, so CALIBRATION_THRESHOLD sits above its 0.05 anchor.
DISCRIMINATION_THRESHOLD = 0.01443
CALIBRATION_THRESHOLD = 0.18237


def deficit(reference: float, imbalanced: float) -> float:
    """Higher-is-better imbalance-induced deficit: D_M = M(balanced CE) - M(imbalanced CE)."""
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
    """Opens when CE's paired BA deficit >= DISCRIMINATION_THRESHOLD and its 95% CI excludes zero."""
    return ba_deficit >= DISCRIMINATION_THRESHOLD and ci_excludes_zero(*ci)


def calibration_gate(tail_nll_deficit: float, ci: tuple[float, float]) -> bool:
    """Opens when CE's tail-group macro-NLL deficit >= CALIBRATION_THRESHOLD nats and its 95% CI excludes zero."""
    return tail_nll_deficit >= CALIBRATION_THRESHOLD and ci_excludes_zero(*ci)


def confidence_interval(dist: np.ndarray) -> tuple[float, float]:
    """95% percentile CI over the bootstrap replicates (index 0 is the observed point)."""
    replicates = dist[1:] if len(dist) > 1 else dist
    return (
        float(np.nanpercentile(replicates, 2.5)),
        float(np.nanpercentile(replicates, 97.5)),
    )


def _gate_comparison(
    severity: str, gate: str, b_dist: np.ndarray, s_dist: np.ndarray, gate_fn: Any
) -> tuple[dict[str, Any], bool, np.ndarray]:
    deficit_dist = b_dist - s_dist
    effect = float(deficit_dist[0])  # replicate 0 is the observed cohort
    ci = confidence_interval(deficit_dist)
    # The gate and the reported effect use the observed-data deficit; the
    # bootstrap replicates supply only the confidence interval.
    passed = gate_fn(effect, ci)
    comparison = {
        "method": "ce",
        "gate": gate,
        "severity": severity,
        "effect": effect,
        "ci": ci,
        "gate_passed": passed,
        "p_value": None,
        "bootstrap_effect": deficit_dist.tolist(),
    }
    return comparison, passed, deficit_dist


def discrimination_gate_comparison(
    severity: str, balanced_ba: np.ndarray, severity_ba: np.ndarray
) -> tuple[dict[str, Any], bool, np.ndarray]:
    """CE-only discrimination-axis deficit and gate check for one severity."""
    return _gate_comparison(
        severity, "discrimination", balanced_ba, severity_ba, discrimination_gate
    )


def calibration_gate_comparison(
    severity: str,
    balanced_tail_nll: np.ndarray | None,
    severity_tail_nll: np.ndarray | None,
) -> tuple[dict[str, Any], bool, np.ndarray] | None:
    """CE-only calibration-axis (tail macro NLL) deficit and gate check for one severity."""
    if balanced_tail_nll is None or severity_tail_nll is None:
        return None
    return _gate_comparison(
        severity, "calibration", severity_tail_nll, balanced_tail_nll, calibration_gate
    )


class _SeverityInputs:
    def __init__(
        self,
        paths: dict[str, Path],
        severity: str,
        balanced: dict[str, Any],
        severity_ce: dict[str, Any],
        ctx: BootstrapContext,
        n_classes: int,
        n_perm: int,
        seed: int,
        assignment: str,
        descriptive_only: bool = False,
    ) -> None:
        self.paths = paths
        self.severity = severity
        self.balanced = balanced
        self.severity_ce = severity_ce
        self.ctx = ctx
        self.n_classes = n_classes
        self.n_perm = n_perm
        self.seed = seed
        self.assignment = assignment
        self.descriptive_only = descriptive_only


def _recovery_comparison(
    inp: _SeverityInputs,
    method: str,
    gate: str,
    effect_dist: np.ndarray,
    deficit_dist: np.ndarray,
    gate_passed: bool,
    p_value: float | None,
) -> dict[str, Any]:
    """Observed effect and recovery ratio (numerator/denominator at replicate 0), CIs."""
    effect_point, deficit_point = float(effect_dist[0]), float(deficit_dist[0])
    with np.errstate(divide="ignore", invalid="ignore"):
        recovery_dist = np.where(deficit_dist != 0, effect_dist / deficit_dist, np.nan)
    return {
        "method": method,
        "gate": gate,
        "severity": inp.severity,
        "effect": effect_point,
        "numerator": effect_point,
        "denominator": deficit_point,
        "ci": confidence_interval(effect_dist),
        "recovery": recovery(effect_point, 0.0, deficit_point),
        "recovery_ci": confidence_interval(recovery_dist),
        "assignment": inp.assignment,
        "gate_passed": gate_passed,
        "p_value": p_value,
        "bootstrap_effect": effect_dist.tolist(),
        "bootstrap_numerator": effect_dist.tolist(),
        "bootstrap_denominator": deficit_dist.tolist(),
    }


def _method_discrimination_recovery(
    inp: _SeverityInputs,
    ba_deficit_dist: np.ndarray,
    severity_ba: np.ndarray,
    method: str,
    method_rec: dict[str, Any],
    gate_passed: bool,
) -> dict[str, Any]:
    """One method's discrimination-axis recovery ratio and permutation p-value."""
    method_ba = inp.ctx.ba_distribution(
        inp.balanced["labels"], method_rec["preds"], inp.n_classes
    )
    p_val = (
        paired_block_permutation_ba(
            inp.balanced["labels"],
            method_rec["preds"],
            inp.severity_ce["preds"],
            inp.ctx.case_ids,
            inp.n_classes,
            inp.n_perm,
            inp.seed,
        )
        if gate_passed and method in PRIMARY_METHODS
        else None
    )
    return _recovery_comparison(
        inp,
        method,
        "discrimination",
        method_ba - severity_ba,
        ba_deficit_dist,
        gate_passed,
        p_val,
    )


def _method_calibration_recovery(
    inp: _SeverityInputs,
    cal_deficit_dist: np.ndarray,
    severity_tail_nll: np.ndarray,
    tail_classes: list[int],
    method: str,
    method_rec: dict[str, Any],
    gate_passed: bool,
) -> dict[str, Any]:
    """One method's calibration-axis (tail NLL) recovery ratio and permutation p-value."""
    method_tail_nll = inp.ctx.tail_nll_distribution(
        inp.balanced["labels"], method_rec["probs"], tail_classes
    )
    p_val = (
        paired_block_permutation_tail_nll(
            inp.balanced["labels"],
            method_rec["probs"],
            inp.severity_ce["probs"],
            inp.ctx.case_ids,
            tail_classes,
            inp.n_perm,
            inp.seed,
        )
        if gate_passed and method in PRIMARY_METHODS
        else None
    )
    return _recovery_comparison(
        inp,
        method,
        "calibration",
        severity_tail_nll - method_tail_nll,
        cal_deficit_dist,
        gate_passed,
        p_val,
    )
