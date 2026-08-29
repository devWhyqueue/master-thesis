from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from imbalance_benchmark.analysis.inference.context import BootstrapContext

__all__ = [
    "DISCRIMINATION_THRESHOLDS",
    "CALIBRATION_THRESHOLDS",
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
]

# Report §"Imbalance deficit, recovery, and inference": the two co-primary gate
# thresholds, one pair per dataset. Derived per PLAN_3 §2 (see
# derive_deficit_thresholds.py's logged "paste into gates.py" lines) from each
# dataset's own seed dispersion -- a single global max would calibrate every
# dataset to the noisiest one (BRACS) and under-gate the stable ones.
DISCRIMINATION_THRESHOLDS: dict[str, float] = {
    "bracs": 0.01443,
    "camelyon16": 0.01443,
    "panda": 0.01443,
    "tcga_ut": 0.01443,
}
CALIBRATION_THRESHOLDS: dict[str, float] = {
    "bracs": 0.18237,
    "camelyon16": 0.18237,
    "panda": 0.18237,
    "tcga_ut": 0.18237,
}


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


# Pre-plan-05 pooled value, kept only as the fallback when a caller has no
# dataset to look up (e.g. a test harness with no config); every real
# pipeline call site threads its own dataset name and uses the table above.
_FALLBACK_DISCRIMINATION_THRESHOLD = 0.01443
_FALLBACK_CALIBRATION_THRESHOLD = 0.18237


def discrimination_gate(
    ba_deficit: float, ci: tuple[float, float], dataset: str | None = None
) -> bool:
    """Opens when CE's paired BA deficit >= this dataset's threshold and its 95% CI excludes zero."""
    threshold = DISCRIMINATION_THRESHOLDS.get(
        dataset or "", _FALLBACK_DISCRIMINATION_THRESHOLD
    )
    return ba_deficit >= threshold and ci_excludes_zero(*ci)


def calibration_gate(
    tail_nll_deficit: float, ci: tuple[float, float], dataset: str | None = None
) -> bool:
    """Opens when CE's tail-group macro-NLL deficit >= this dataset's threshold (nats) and its 95% CI excludes zero."""
    threshold = CALIBRATION_THRESHOLDS.get(
        dataset or "", _FALLBACK_CALIBRATION_THRESHOLD
    )
    return tail_nll_deficit >= threshold and ci_excludes_zero(*ci)


def confidence_interval(dist: np.ndarray) -> tuple[float, float]:
    """95% percentile CI over the bootstrap replicates (index 0 is the observed point)."""
    replicates = dist[1:] if len(dist) > 1 else dist
    return (
        float(np.nanpercentile(replicates, 2.5)),
        float(np.nanpercentile(replicates, 97.5)),
    )


def _gate_comparison(
    severity: str,
    gate: str,
    b_dist: np.ndarray,
    s_dist: np.ndarray,
    dataset: str | None,
    gate_fn: Any,
) -> tuple[dict[str, Any], bool, np.ndarray]:
    deficit_dist = b_dist - s_dist
    effect = float(deficit_dist[0])  # replicate 0 is the observed cohort
    ci = confidence_interval(deficit_dist)
    # The gate and the reported effect use the observed-data deficit; the
    # bootstrap replicates supply only the confidence interval.
    passed = gate_fn(effect, ci, dataset)
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
    severity: str,
    balanced_ba: np.ndarray,
    severity_ba: np.ndarray,
    dataset: str | None = None,
) -> tuple[dict[str, Any], bool, np.ndarray]:
    """CE-only discrimination-axis deficit and gate check for one severity."""
    return _gate_comparison(
        severity,
        "discrimination",
        balanced_ba,
        severity_ba,
        dataset,
        discrimination_gate,
    )


def calibration_gate_comparison(
    severity: str,
    balanced_tail_nll: np.ndarray | None,
    severity_tail_nll: np.ndarray | None,
    dataset: str | None = None,
) -> tuple[dict[str, Any], bool, np.ndarray] | None:
    """CE-only calibration-axis (tail macro NLL) deficit and gate check for one severity."""
    if balanced_tail_nll is None or severity_tail_nll is None:
        return None
    return _gate_comparison(
        severity,
        "calibration",
        severity_tail_nll,
        balanced_tail_nll,
        dataset,
        calibration_gate,
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
        dataset: str = "",
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
        self.dataset = dataset
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
