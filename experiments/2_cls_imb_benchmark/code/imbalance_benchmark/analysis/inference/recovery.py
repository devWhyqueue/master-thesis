from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from imbalance_benchmark.analysis.inference.context import (
    Baseline,
    BootstrapContext,
    balanced_baseline,
)
from imbalance_benchmark.analysis.inference.gates import (
    calibration_gate_comparison,
    confidence_interval,
    discrimination_gate_comparison,
)
from imbalance_benchmark.analysis.inference.permutation import (
    paired_block_permutation_ba,
    paired_block_permutation_tail_nll,
)
from imbalance_benchmark.analysis.query import load_seed_predictions

__all__ = ["gates_and_recovery"]


@dataclass
class _SeverityInputs:
    """Everything one severity's gate/recovery comparisons need, bundled to keep call sites short."""

    paths: dict[str, Path]
    severity: str
    balanced: dict[str, Any]
    severity_ce: dict[str, Any]
    ctx: BootstrapContext
    n_classes: int
    n_perm: int
    seed: int
    assignment: str


def _recovery_comparison(
    method: str, gate: str, severity: str, effect_dist: np.ndarray, p_value: float
) -> dict[str, Any]:
    """One method's bootstrap recovery ratio and paired permutation p-value for one gate."""
    return {
        "method": method,
        "gate": gate,
        "severity": severity,
        "effect": float(np.nanmean(effect_dist)),
        "ci": confidence_interval(effect_dist),
        "gate_passed": True,
        "p_value": p_value,
    }


def _method_discrimination_recovery(
    inp: _SeverityInputs,
    ba_deficit_dist: np.ndarray,
    severity_ba: np.ndarray,
    method: str,
    method_rec: dict[str, Any],
) -> dict[str, Any]:
    """One method's discrimination-axis recovery ratio and permutation p-value against imbalanced CE."""
    method_ba = inp.ctx.ba_distribution(
        inp.balanced["labels"], method_rec["preds"], inp.n_classes
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        recovery_dist = np.where(
            ba_deficit_dist != 0, (method_ba - severity_ba) / ba_deficit_dist, np.nan
        )
    p_val = paired_block_permutation_ba(
        inp.balanced["labels"],
        method_rec["preds"],
        inp.severity_ce["preds"],
        inp.ctx.case_ids,
        inp.n_classes,
        inp.n_perm,
        inp.seed,
    )
    return _recovery_comparison(
        method, "discrimination", inp.severity, recovery_dist, p_val
    )


def _method_calibration_recovery(
    inp: _SeverityInputs,
    cal_deficit_dist: np.ndarray,
    severity_tail_nll: np.ndarray,
    tail_classes: list[int],
    method: str,
    method_rec: dict[str, Any],
) -> dict[str, Any]:
    """One method's calibration-axis (tail NLL) recovery ratio and permutation p-value."""
    method_tail_nll = inp.ctx.tail_nll_distribution(
        inp.balanced["labels"], method_rec["probs"], tail_classes
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        rec_cal_dist = np.where(
            cal_deficit_dist != 0,
            (severity_tail_nll - method_tail_nll) / cal_deficit_dist,
            np.nan,
        )
    p_val = paired_block_permutation_tail_nll(
        inp.balanced["labels"],
        method_rec["probs"],
        inp.severity_ce["probs"],
        inp.ctx.case_ids,
        tail_classes,
        inp.n_perm,
        inp.seed,
    )
    return _recovery_comparison(
        method, "calibration", inp.severity, rec_cal_dist, p_val
    )


def _method_recoveries(
    inp: _SeverityInputs,
    disc_gate: bool,
    cal_gate: bool,
    ba_deficit_dist: np.ndarray,
    cal_deficit_dist: np.ndarray | None,
    severity_ba: np.ndarray,
    severity_tail_nll: np.ndarray | None,
    tail_classes: list[int],
) -> list[dict[str, Any]]:
    """Every non-CE method's discrimination/calibration recovery in one severity condition."""
    out: list[dict[str, Any]] = []
    results_dir = inp.paths["results"] / f"assignment={inp.assignment}" / inp.severity
    if not results_dir.exists():
        results_dir = inp.paths["results"] / inp.severity
    if not results_dir.exists():
        return out
    for method_dir in sorted(results_dir.iterdir()):
        method = method_dir.name
        if method == "ce":
            continue
        method_rec = load_seed_predictions(inp.paths, inp.severity, method, inp.assignment)
        if method_rec is None:
            continue
        if disc_gate:
            out.append(
                _method_discrimination_recovery(
                    inp, ba_deficit_dist, severity_ba, method, method_rec
                )
            )
        if cal_gate and cal_deficit_dist is not None and severity_tail_nll is not None:
            out.append(
                _method_calibration_recovery(
                    inp,
                    cal_deficit_dist,
                    severity_tail_nll,
                    tail_classes,
                    method,
                    method_rec,
                )
            )
    return out


def _severity_comparisons(
    inp: _SeverityInputs,
    balanced_ba: np.ndarray,
    balanced_tail_nll: np.ndarray | None,
    tail_classes: list[int],
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
    comparisons = [disc_comparison]
    cal_gate, cal_deficit_dist = False, None
    cal_result = calibration_gate_comparison(
        inp.severity, balanced_tail_nll, severity_tail_nll
    )
    if cal_result is not None:
        cal_comparison, cal_gate, cal_deficit_dist = cal_result
        comparisons.append(cal_comparison)
    if disc_gate or cal_gate:
        comparisons += _method_recoveries(
            inp,
            disc_gate,
            cal_gate,
            ba_deficit_dist,
            cal_deficit_dist,
            severity_ba,
            severity_tail_nll,
            tail_classes,
        )
    return comparisons


def _severity_result(
    baseline: Baseline, paths: dict[str, Path], severity: str, seed: int, assignment: str
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
    )
    return _severity_comparisons(
        inp, baseline.ba, baseline.tail_nll, baseline.tail_classes
    )


def gates_and_recovery(
    paths: dict[str, Path],
    config: dict[str, Any],
    freeze: dict[str, Any],
    n_replicates: int,
    seed: int,
) -> list[dict[str, Any]]:
    """CE-only deficit gates, then bootstrap recovery + paired permutation p-values per method.

    Both bootstrap and permutation calculations retain the complete matched
    confirmation-seed block; no representative seed is substituted.
    """
    comparisons: list[dict[str, Any]] = []
    for assignment in freeze.get("tail_assignments", {"native": []}):
        baseline = balanced_baseline(paths, config, freeze, n_replicates, seed, assignment)
        if baseline is None:
            continue
        for severity in ("moderate", "severe"):
            comparisons += _severity_result(baseline, paths, severity, seed, assignment)
    return comparisons
