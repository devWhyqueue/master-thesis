from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

from imbalance_benchmark.analysis.calibration import (
    apply_temperature,
    fit_temperature,
    reliability_curve,
)
from imbalance_benchmark.analysis.db import discover_result_dirs, ingest_run
from imbalance_benchmark.analysis.inference.holm import apply_holm
from imbalance_benchmark.analysis.metrics import (
    assign_tiers,
    brier_score,
    expected_calibration_error,
    negative_log_likelihood,
)
from imbalance_benchmark.common import read_run_record, write_json

__all__ = ["ingest_all_runs", "calibration_summary", "write_diagnostics"]


def ingest_all_runs(
    conn: sqlite3.Connection, paths: dict[str, Path], freeze: dict[str, Any]
) -> None:
    """Ingest every confirmed run under results/, tier-annotated from the freeze manifest."""
    for condition, method, seed_idx, result_dir in discover_result_dirs(
        paths["results"]
    ):
        record = read_run_record(result_dir)
        if record is None:
            continue
        _ingest_discovered_run(
            conn, freeze, condition, method, seed_idx, result_dir, record
        )


def _ingest_discovered_run(
    conn: sqlite3.Connection,
    freeze: dict[str, Any],
    condition: str,
    method: str,
    seed_idx: int,
    result_dir: Path,
    record: dict[str, Any],
) -> None:
    """Add one run with tiers derived from its frozen assignment allocation."""
    provenance = record.get("provenance", {})
    if provenance.get("freeze_content_sha256") != freeze.get("content_sha256"):
        raise RuntimeError("Run belongs to a stale manifest freeze; regenerate it")
    class_names = record.get("class_names", [])
    assignment = record.get("assignment", "native")
    if condition not in {"natural", "balanced"} and assignment not in freeze.get(
        "tail_assignments", {}
    ):
        raise RuntimeError("Run assignment is absent from the current manifest freeze")
    allocated = (
        freeze.get("conditions", {}).get(condition, {}).get("allocated_counts", {})
    )
    if not allocated:
        allocated = (
            freeze.get("assignment_conditions", {})
            .get(assignment, {})
            .get(condition, {})
            .get("allocated_counts", {})
        )
    tiers = {}
    if condition != "balanced" and class_names and allocated:
        tiers = assign_tiers(
            class_names,
            allocated,
            freeze.get("tail_assignments", {}).get(assignment, class_names),
        )
    run_id = f"{record.get('benchmark', 'unknown')}:{assignment}:{condition}:{method}:seed={seed_idx}"
    ingest_run(conn, run_id, result_dir, (condition, method, seed_idx), record, tiers)


def write_diagnostics(
    paths: dict[str, Path], comparisons: list[dict[str, Any]]
) -> None:
    """Persist gate/recovery and calibration diagnostics for one analyzed split."""
    write_json(
        paths["data"] / "gates_and_recovery.json",
        {"comparisons": apply_holm(comparisons)},
    )
    write_json(paths["data"] / "calibration_summary.json", calibration_summary(paths))


def _run_calibration(record: dict[str, Any]) -> dict[str, Any] | None:
    """Fit temperature on one run's validation logits; report raw vs. calibrated test NLL."""
    if "validation" not in record["splits"] or "test" not in record["splits"]:
        return None
    val, test = record["splits"]["validation"], record["splits"]["test"]
    val_logits = np.array(val.get("target_prior_logits", val["logits"]))
    test_logits = np.array(test.get("target_prior_logits", test["logits"]))
    fit = fit_temperature(val_logits, np.array(val["labels"]))
    labels = np.array(test["labels"])
    raw_probs = np.array(test.get("probabilities", []))
    calibrated_probs = apply_temperature(test_logits, fit.temperature)
    centers, confidence, accuracy = reliability_curve(calibrated_probs, labels)
    n_classes = calibrated_probs.shape[1]
    return {
        "temperature": fit.temperature,
        "raw_test_nll": negative_log_likelihood(
            labels,
            np.array(test.get("raw_probabilities", raw_probs)),
        ),
        "target_prior_test_nll": negative_log_likelihood(labels, raw_probs),
        "temperature_scaled_test_nll": negative_log_likelihood(
            labels, calibrated_probs
        ),
        "temperature_scaled_test_brier": brier_score(
            labels, calibrated_probs, n_classes
        ),
        "temperature_scaled_test_ece": expected_calibration_error(
            labels, calibrated_probs
        ),
        "temperature_scaled_test_ece_ci": test.get("temperature_scaled_ece_ci"),
        "temperature_scaled_reliability": {
            "bin_centers": centers.tolist(),
            "mean_confidence": confidence.tolist(),
            "accuracy": accuracy.tolist(),
        },
    }


def calibration_summary(paths: dict[str, Path]) -> dict[str, Any]:
    """Fit temperature per run on recorded validation logits; report raw vs. calibrated test NLL.

    Applies the target-prior correction first for the two methods that define
    one (Eq. posthoc/train-time-target-prior), matching the report's "fitted
    ... after the appropriate target-prior correction".
    """
    summary: dict[str, Any] = {}
    for condition, method, seed_idx, result_dir in discover_result_dirs(
        paths["results"]
    ):
        record = read_run_record(result_dir)
        if record is None:
            continue
        entry = _run_calibration(record)
        if entry is not None:
            assignment = record.get("assignment", "native")
            run_id = f"{record.get('benchmark', 'unknown')}:{assignment}:{condition}:{method}:seed={seed_idx}"
            summary[run_id] = entry
    return summary
