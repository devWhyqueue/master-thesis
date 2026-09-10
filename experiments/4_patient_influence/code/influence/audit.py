"""Preflight audit: cell integrity, matched class counts, baseline reuse, and
patient-contribution inequality, for every split and support condition.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from decodability.audit import _verify_cell_audit
from decodability.evidence import CellEvidence, load_cell
from imbalance_benchmark.common import output_root, sign_file

from influence import INPUT_DIM, SUPPORTS
from influence.baseline import verify_baseline
from influence.weights import contribution_audit, patient_average_weights

__all__ = ["run_preflight"]

logger = logging.getLogger(__name__)


def _verify_matched_class_counts(cells: dict[str, CellEvidence]) -> None:
    """Verify class-specific train patch counts match across support conditions."""
    n_classes = len(cells[SUPPORTS[0]].class_names)
    counts = {
        support: np.bincount(cell.train_y, minlength=n_classes)
        for support, cell in cells.items()
    }
    reference = counts[SUPPORTS[0]]
    for support in SUPPORTS[1:]:
        if not np.array_equal(reference, counts[support]):
            raise RuntimeError(
                f"Class patch counts differ between {SUPPORTS[0]} ({reference.tolist()}) "
                f"and {support} ({counts[support].tolist()})"
            )


def _audit_cell(
    config: dict[str, Any], split_index: int, support: str, cell: CellEvidence
) -> dict[str, Any]:
    """Audit one cell: integrity, baseline reuse, and contribution inequality."""
    weights = patient_average_weights(cell.train_y, cell.train_patients)
    return {
        "cell_stats": _verify_cell_audit(cell),
        "baseline": verify_baseline(config, split_index, support, cell),
        "contribution_audit": contribution_audit(
            cell.train_y, cell.train_patients, cell.class_names
        ),
        "weight_sum": float(np.sum(weights)),
    }


def _audit_split(config: dict[str, Any], split_index: int) -> dict[str, Any]:
    """Audit both support conditions for one split."""
    cells = {s: load_cell(config, split_index, s) for s in SUPPORTS}
    _verify_matched_class_counts(cells)
    return {
        support: _audit_cell(config, split_index, support, cell)
        for support, cell in cells.items()
    }


def run_preflight(config: dict[str, Any]) -> Path:
    """Run the preflight audit across all 3 splits and supports."""
    report = {
        "status": "pass",
        "input_dim": INPUT_DIM,
        "splits": {str(i): _audit_split(config, i) for i in range(3)},
    }
    out_p = output_root(config) / "data" / "preflight.json"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(report, indent=2), encoding="utf-8")
    sign_file(out_p)
    logger.info("Preflight signed at %s", out_p)
    return out_p
