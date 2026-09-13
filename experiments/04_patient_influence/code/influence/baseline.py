"""Locate, verify, and load exp-3's reused patch-average logistic baselines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from decodability import allocation_manifest, exp2_split_paths
from decodability.evidence import CellEvidence
from imbalance_benchmark.analysis.query import read_run_record
from imbalance_benchmark.common import compute_sha256, verify_signed_file

from influence import MAX_ITER, TOLERANCE, exp3_root, inherited_lambda

__all__ = ["verify_baseline", "load_baseline_predictions"]


def _baseline_dir(config: dict[str, Any], split_index: int, support: str) -> Path:
    """Resolve exp-3's baseline result directory for the inherited lambda."""
    _, param_str = inherited_lambda(config, support)
    return (
        exp3_root(config)
        / f"split={split_index}"
        / "results"
        / support
        / "logreg"
        / param_str
    )


def _load_baseline_record(
    config: dict[str, Any], split_index: int, support: str
) -> dict[str, Any]:
    """Load exp-3's test-split baseline record, raising if it is missing."""
    baseline_dir = _baseline_dir(config, split_index, support)
    record = read_run_record(
        baseline_dir, splits=("test",), array_fields=("labels", "preds")
    )
    if record is None or "test" not in record.get("splits", {}):
        raise RuntimeError(f"Missing exp-3 baseline test record at {baseline_dir}")
    return record


def _verify_manifest_identity(
    config: dict[str, Any], split_index: int, support: str
) -> None:
    """Compare exp-3's signed preflight manifest hash to exp-4's own manifest."""
    preflight_path = exp3_root(config) / "data" / "preflight.json"
    verify_signed_file(preflight_path)
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    exp3_sha = (
        preflight.get("splits", {})
        .get(str(split_index), {})
        .get(support, {})
        .get("manifest_sha256")
    )
    exp2_paths = exp2_split_paths(config, split_index)
    own_sha = compute_sha256(allocation_manifest(exp2_paths, support))
    if exp3_sha != own_sha:
        raise RuntimeError(
            f"manifest_sha256 mismatch for split {split_index}, support {support}: "
            f"exp-3 preflight {exp3_sha} != exp-4 manifest {own_sha}"
        )


def _verify_solver(solver: dict[str, Any], lam: float, n_train: int) -> None:
    """Verify regularization, solver settings, and convergence match exp-3's fit."""
    if not np.isclose(solver.get("lambda", np.nan), lam):
        raise RuntimeError(f"Baseline solver.lambda {solver.get('lambda')} != {lam}")
    expected_c = 1.0 / (n_train * lam)
    if not np.isclose(solver.get("C", np.nan), expected_c, rtol=1e-6):
        raise RuntimeError(f"Baseline solver.C {solver.get('C')} != {expected_c}")
    if solver.get("solver") != "lbfgs" or solver.get("precision") != "float64":
        raise RuntimeError(f"Baseline solver settings unexpected: {solver}")
    if not np.isclose(solver.get("tolerance", np.nan), TOLERANCE):
        raise RuntimeError(
            f"Baseline solver.tolerance {solver.get('tolerance')} != {TOLERANCE}"
        )
    if not np.isclose(solver.get("solver_tolerance", np.nan), TOLERANCE * n_train):
        raise RuntimeError(
            f"Baseline solver.solver_tolerance {solver.get('solver_tolerance')} "
            f"!= {TOLERANCE * n_train}"
        )
    if solver.get("max_iter") != MAX_ITER:
        raise RuntimeError(
            f"Baseline solver.max_iter {solver.get('max_iter')} != {MAX_ITER}"
        )
    if solver.get("converged") is not True:
        raise RuntimeError("Baseline solver did not converge")


def _verify_predictions(preds: np.ndarray, n_test: int, n_classes: int) -> None:
    """Verify prediction alignment: presence, finiteness, length, and class range."""
    if len(preds) != n_test:
        raise RuntimeError(f"Baseline prediction length {len(preds)} != {n_test}")
    if not np.all(np.isfinite(preds)):
        raise RuntimeError("Baseline predictions contain non-finite values")
    if np.any(preds < 0) or np.any(preds >= n_classes):
        raise RuntimeError(f"Baseline predictions outside class range [0, {n_classes})")


def _verify_labels_match(labels: np.ndarray, test_y: np.ndarray, support: str) -> None:
    """Verify baseline test labels match the locked cell's test labels."""
    if len(labels) != len(test_y) or not np.array_equal(labels, test_y):
        raise RuntimeError(
            f"Baseline test labels do not match locked cell for {support}"
        )


def verify_baseline(
    config: dict[str, Any], split_index: int, support: str, cell: CellEvidence
) -> dict[str, Any]:
    """Verify exp-3's patch-average baseline is reusable for one split and support."""
    lam, param_str = inherited_lambda(config, support)
    record = _load_baseline_record(config, split_index, support)

    if record.get("feature_extraction", {}) != config.get("feature_extraction", {}):
        raise RuntimeError(f"Baseline feature_extraction mismatch for {support}")
    _verify_manifest_identity(config, split_index, support)

    test_data = record["splits"]["test"]
    _verify_labels_match(np.asarray(test_data["labels"]), cell.test_y, support)

    _verify_solver(record.get("solver", {}), lam, len(cell.train_y))
    preds = np.asarray(test_data["preds"])
    _verify_predictions(preds, len(cell.test_y), len(cell.class_names))

    return {
        "support": support,
        "param": param_str,
        "lambda": lam,
        "n_test": len(cell.test_y),
        "converged": True,
    }


def load_baseline_predictions(
    config: dict[str, Any], split_index: int, support: str
) -> tuple[np.ndarray, np.ndarray]:
    """Load exp-3's baseline test labels and predictions for one split and support."""
    record = _load_baseline_record(config, split_index, support)
    test_data = record["splits"]["test"]
    return np.asarray(test_data["labels"]), np.asarray(test_data["preds"])
