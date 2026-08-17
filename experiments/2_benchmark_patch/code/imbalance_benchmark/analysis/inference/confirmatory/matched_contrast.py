"""The matched-vs-unmatched contrast row (protocol app:testing)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from imbalance_benchmark.analysis.inference.gates import confidence_interval
from imbalance_benchmark.analysis.inference.confirmatory.holm import (
    MATCHED_CONTRAST_METHOD,
)
from imbalance_benchmark.analysis.predictors.rq3_analysis import _rq3_group

__all__ = ["load_matching_units", "contrast_rows"]

logger = logging.getLogger(__name__)


def load_matching_units(
    paths: dict[str, Path], freeze: dict[str, Any]
) -> dict[tuple[str, str], Any]:
    """Load this dataset-regime's matching-record units, keyed by (assignment, severity).

    Absent for datasets analyzed before ``match`` was run, or when the frozen
    dataset provenance lacks a target name; both leave the contrast disabled
    rather than failing analysis.
    """
    matching_path = paths["root"].parent / "data" / "matching_record.json"
    target = freeze.get("dataset_provenance", {}).get("target")
    if not matching_path.exists() or not isinstance(target, str) or not target.strip():
        return {}
    group = _rq3_group(freeze)
    units = json.loads(matching_path.read_text()).get("units", {})
    return {
        (unit["assignment"], unit["severity"]): unit
        for unit in units.values()
        if unit.get("group") == group
    }


def _matched_vs_unmatched_row(
    severity: str,
    gate: str,
    gate_passed: bool,
    bootstrap_by_method: dict[str, np.ndarray],
    matched_methods: list[str],
    unmatched_methods: list[str],
    dominant: str | None,
) -> dict[str, Any] | None:
    """One dataset's paired matched-vs-unmatched effect (protocol app:testing)."""
    missing = [
        method
        for method in (*matched_methods, *unmatched_methods)
        if method not in bootstrap_by_method
    ]
    if missing:
        logger.warning(
            "recovery: matched_vs_unmatched %s/%s skipped, missing predictions for %s",
            severity,
            gate,
            missing,
        )
        return None
    matched = np.mean([bootstrap_by_method[m] for m in matched_methods], axis=0)
    unmatched = np.mean([bootstrap_by_method[m] for m in unmatched_methods], axis=0)
    contrast = matched - unmatched
    return {
        "method": MATCHED_CONTRAST_METHOD,
        "gate": gate,
        "severity": severity,
        "effect": float(contrast[0]),
        "ci": confidence_interval(contrast),
        "gate_passed": gate_passed,
        "p_value": None,
        "bootstrap_effect": contrast.tolist(),
        "dominant": dominant,
        "matched_methods": list(matched_methods),
        "unmatched_methods": list(unmatched_methods),
    }


def contrast_rows(
    severity: str,
    disc_gate: bool,
    cal_gate: bool,
    disc_bootstrap: dict[str, np.ndarray],
    cal_bootstrap: dict[str, np.ndarray],
    matching_unit: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """The matched-vs-unmatched contrast row(s), if this unit has a non-ambiguous label."""
    if matching_unit is None or matching_unit.get("ambiguous", True):
        return []
    matched, unmatched, dominant = (
        matching_unit["matched_methods"],
        matching_unit["unmatched_methods"],
        matching_unit["dominant"],
    )
    axes = [("discrimination", disc_gate, disc_bootstrap)]
    if cal_bootstrap:
        axes.append(("calibration", cal_gate, cal_bootstrap))
    rows = (
        _matched_vs_unmatched_row(
            severity, gate, gate_passed, bootstrap, matched, unmatched, dominant
        )
        for gate, gate_passed, bootstrap in axes
    )
    return [row for row in rows if row is not None]
