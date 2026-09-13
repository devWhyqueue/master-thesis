"""Preflight audit: verify freezes, manifests, disjointness, and MLP reference."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from imbalance_benchmark.analysis.query import load_seed_predictions
from imbalance_benchmark.common import (
    compute_sha256,
    output_root,
    sign_file,
)

from decodability import (
    INPUT_DIM,
    MIN_TRAIN_PATCHES,
    MLP_ASSIGNMENT,
    SUPPORTS,
    allocation_manifest,
    exp2_split_paths,
)
from decodability.evidence import load_cell

__all__ = ["run_preflight"]

logger = logging.getLogger(__name__)


def _verify_cell_audit(cell: Any) -> dict[str, Any]:
    """Audit one cell's features, patients, and labels."""
    train_cases = set(cell.train_patients)
    val_cases = set(cell.val_identity["case_id"].astype(str))
    test_cases = set(cell.test_identity["case_id"].astype(str))

    train_val = train_cases & val_cases
    train_test = train_cases & test_cases
    val_test = val_cases & test_cases
    if train_val or train_test or val_test:
        raise RuntimeError(
            f"Patient overlap detected in {cell.support}: "
            f"train_val={len(train_val)}, train_test={len(train_test)}, "
            f"val_test={len(val_test)}"
        )

    n_train = len(cell.train_y)
    if n_train < MIN_TRAIN_PATCHES:
        raise RuntimeError(
            f"Train patch count {n_train} < {MIN_TRAIN_PATCHES} for {cell.support}"
        )

    train_mat = cell.train_x.cpu().numpy()
    if not np.all(np.isfinite(train_mat)):
        raise RuntimeError(f"Non-finite train features in {cell.support}")

    norms = np.linalg.norm(train_mat, axis=1)
    if np.any(norms <= 0.0):
        raise RuntimeError(f"Zero L2-norm feature vector in {cell.support}")

    classes, counts = np.unique(cell.train_y, return_counts=True)
    return {
        "n_train": n_train,
        "n_classes": len(cell.class_names),
        "class_counts": {str(c): int(cnt) for c, cnt in zip(classes, counts)},
        "n_patients_train": len(train_cases),
        "n_patients_val": len(val_cases),
        "n_patients_test": len(test_cases),
    }


def _audit_mlp_reference(
    exp2_paths: dict[str, Path], support: str, test_y: np.ndarray
) -> dict[str, Any]:
    """Verify complete 5-run confirmation block for MLP reference."""
    assignment = MLP_ASSIGNMENT[support]
    stacked = load_seed_predictions(
        exp2_paths, support, "ce", assignment=assignment, fields=("preds",)
    )
    if stacked is None:
        raise RuntimeError(
            f"Missing complete 5-seed MLP reference for {support} ({assignment})"
        )
    preds = stacked["preds"]
    stored_labels = stacked["labels"]
    if preds.shape[0] != 5:
        raise RuntimeError(f"Expected 5 MLP runs, got {preds.shape[0]}")
    if preds.shape[1] != len(test_y):
        raise RuntimeError(
            f"MLP prediction length {preds.shape[1]} != test size {len(test_y)}"
        )
    if not np.array_equal(stored_labels, test_y):
        raise RuntimeError("Stored MLP labels do not match locked test labels")
    return {"seeds": 5, "n_test": len(test_y)}


def _audit_split(config: dict[str, Any], split_index: int) -> dict[str, Any]:
    """Audit both supports for one split."""
    exp2_paths = exp2_split_paths(config, split_index)
    split_rep: dict[str, Any] = {}
    for supp in SUPPORTS:
        man_p = allocation_manifest(exp2_paths, supp)
        cell = load_cell(config, split_index, supp)
        split_rep[supp] = {
            "manifest_path": str(man_p),
            "manifest_sha256": compute_sha256(man_p),
            "cell_stats": _verify_cell_audit(cell),
            "mlp_reference": _audit_mlp_reference(exp2_paths, supp, cell.test_y),
        }
    return split_rep


def run_preflight(config: dict[str, Any]) -> Path:
    """Run Gate 0 preflight audit across all 3 splits and supports."""
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
