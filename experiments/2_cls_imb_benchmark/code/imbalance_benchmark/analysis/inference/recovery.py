from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from imbalance_benchmark.analysis.inference.context import (
    Baseline,
    balanced_baseline,
)
from imbalance_benchmark.analysis.inference.gates import (
    _SeverityInputs,
    _method_discrimination_recovery,
    _method_calibration_recovery,
    calibration_gate_comparison,
    confidence_interval,
    discrimination_gate_comparison,
)
from imbalance_benchmark.analysis.query import load_seed_predictions
from imbalance_benchmark.modeling.context import roster_for_regime

__all__ = ["gates_and_recovery"]


def _method_recoveries(
    inp: _SeverityInputs,
    disc_gate: bool,
    cal_gate: bool,
    ba_deficit_dist: np.ndarray,
    cal_deficit_dist: np.ndarray | None,
    severity_ba: np.ndarray,
    severity_tail_nll: np.ndarray | None,
    tail_classes: list[int],
    expected_methods: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Every non-CE method's discrimination/calibration recovery in one severity condition."""
    out: list[dict[str, Any]] = []
    for method in expected_methods:
        if method == "ce":
            continue
        method_rec = load_seed_predictions(
            inp.paths, inp.severity, method, inp.assignment
        )
        if method_rec is None:
            continue
        discrimination = _method_discrimination_recovery(
            inp, ba_deficit_dist, severity_ba, method, method_rec, disc_gate
        )
        ece_dist = inp.ctx.ece_distribution(method_rec["labels"], method_rec["probs"])
        discrimination["ece"] = float(np.nanmean(ece_dist))
        discrimination["ece_ci"] = confidence_interval(ece_dist)
        out.append(discrimination)
        if cal_deficit_dist is not None and severity_tail_nll is not None:
            calibration = _method_calibration_recovery(
                inp,
                cal_deficit_dist,
                severity_tail_nll,
                tail_classes,
                method,
                method_rec,
                cal_gate,
            )
            calibration["ece"] = float(np.nanmean(ece_dist))
            calibration["ece_ci"] = confidence_interval(ece_dist)
            out.append(calibration)
    return out


def _severity_comparisons(
    inp: _SeverityInputs,
    balanced_ba: np.ndarray,
    balanced_tail_nll: np.ndarray | None,
    tail_classes: list[int],
    expected_methods: tuple[str, ...],
) -> list[dict[str, Any]]:
    """CE-only gate checks for one severity, then every method's recovery if either gate opened."""
    severity_ba = inp.ctx.ba_distribution(
        inp.severity_ce["labels"], inp.severity_ce["preds"], inp.n_classes
    )
    severity_tail_nll = inp.ctx.tail_nll_distribution(
        inp.severity_ce["labels"], inp.severity_ce["probs"], tail_classes
    )
    disc_comparison, disc_gate, ba_deficit_dist = discrimination_gate_comparison(
        inp.severity, balanced_ba, severity_ba
    )
    ece_dist = inp.ctx.ece_distribution(
        inp.severity_ce["labels"], inp.severity_ce["probs"]
    )
    disc_comparison["ece"] = float(np.nanmean(ece_dist))
    disc_comparison["ece_ci"] = confidence_interval(ece_dist)
    comparisons = [disc_comparison]
    cal_gate, cal_deficit_dist = False, None
    cal_result = calibration_gate_comparison(
        inp.severity, balanced_tail_nll, severity_tail_nll
    )
    if cal_result is not None:
        cal_comparison, cal_gate, cal_deficit_dist = cal_result
        comparisons.append(cal_comparison)
    if inp.descriptive_only:
        disc_gate = cal_gate = False
        for comparison in comparisons:
            comparison["gate_passed"] = False
    comparisons += _method_recoveries(
        inp,
        disc_gate,
        cal_gate,
        ba_deficit_dist,
        cal_deficit_dist,
        severity_ba,
        severity_tail_nll,
        tail_classes,
        expected_methods,
    )
    for comparison in comparisons:
        comparison.setdefault("assignment", inp.assignment)
        comparison["descriptive_only"] = inp.descriptive_only
    return comparisons


def _severity_result(
    baseline: Baseline,
    paths: dict[str, Path],
    severity: str,
    seed: int,
    assignment: str,
    descriptive_only: bool,
    expected_methods: tuple[str, ...],
) -> list[dict[str, Any]]:
    """One severity's gate/recovery comparisons against the shared balanced-CE baseline."""
    severity_ce = load_seed_predictions(paths, severity, "ce", assignment)
    if severity_ce is None:
        return []
    inp = _SeverityInputs(
        paths,
        severity,
        baseline.balanced,
        severity_ce,
        baseline.ctx,
        baseline.n_classes,
        baseline.n_perm,
        seed,
        assignment,
        descriptive_only,
    )
    return _severity_comparisons(
        inp, baseline.ba, baseline.tail_nll, baseline.tail_classes, expected_methods
    )


def gates_and_recovery(
    paths: dict[str, Path],
    config: dict[str, Any],
    freeze: dict[str, Any],
    n_replicates: int,
    seed: int,
) -> list[dict[str, Any]]:
    """CE-only deficit gates, then bootstrap recovery + paired permutation p-values per method."""
    descriptive_only = bool(
        freeze.get("bootstrap_preflight", {}).get("is_descriptive_only", False)
    )
    comparisons: list[dict[str, Any]] = []
    is_mil = config.get("dataset", {}).get("regime", "patch") == "wsi"
    expected_methods = roster_for_regime(is_mil)
    for assignment in freeze.get("tail_assignments", {"native": []}):
        baseline = balanced_baseline(
            paths, config, freeze, n_replicates, seed, assignment
        )
        if baseline is None:
            continue
        for severity in ("moderate", "severe"):
            comparisons += _severity_result(
                baseline,
                paths,
                severity,
                seed,
                assignment,
                descriptive_only,
                expected_methods,
            )
    return comparisons
