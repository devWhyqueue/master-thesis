from __future__ import annotations

import logging
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
from imbalance_benchmark.analysis.inference.confirmatory.holm import apply_holm
from imbalance_benchmark.analysis.metrics import (
    assign_tiers,
    brier_score,
    expected_calibration_error,
    negative_log_likelihood,
)
from imbalance_benchmark.common import read_run_record, write_json
from imbalance_benchmark.manifest.freeze import accepted_freeze_hashes

__all__ = [
    "ingest_all_runs",
    "calibration_summary",
    "method_diagnostics_summary",
    "write_diagnostics",
]

logger = logging.getLogger(__name__)


def ingest_all_runs(
    conn: sqlite3.Connection, paths: dict[str, Path], freeze: dict[str, Any]
) -> None:
    """Rebuild the run table from results/: ingest what's there, prune what's gone.

    A method retired from a condition's roster (or a directory removed after
    a rerun) must not leave its old run_id lingering forever - discover_run_dirs
    only ever upserts what it finds, so anything no longer discovered here is
    explicitly dropped, keeping analyze-combine's DB-derived key set truthful
    to the current results/ tree instead of its entire ingestion history.
    """
    discovered_ids = set()
    for condition, method, seed_idx, result_dir in discover_result_dirs(
        paths["results"]
    ):
        record = read_run_record(result_dir, array_fields=())
        if record is None:
            continue
        discovered_ids.add(
            _ingest_discovered_run(
                conn, freeze, condition, method, seed_idx, result_dir, record
            )
        )
    _prune_stale_runs(conn, discovered_ids)


def _prune_stale_runs(conn: sqlite3.Connection, discovered_ids: set[str]) -> None:
    """Delete run rows no longer backed by a discovered result_dir (cascades evals)."""
    existing = {row[0] for row in conn.execute("SELECT run_id FROM runs")}
    stale = existing - discovered_ids
    if stale:
        conn.executemany(
            "DELETE FROM runs WHERE run_id = ?", [(run_id,) for run_id in stale]
        )
        conn.commit()


def _ingest_discovered_run(
    conn: sqlite3.Connection,
    freeze: dict[str, Any],
    condition: str,
    method: str,
    seed_idx: int,
    result_dir: Path,
    record: dict[str, Any],
) -> str:
    """Add one run with tiers derived from its frozen assignment allocation."""
    provenance = record.get("provenance", {})
    if provenance.get("freeze_content_sha256") not in accepted_freeze_hashes(freeze):
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
    return run_id


def write_diagnostics(
    paths: dict[str, Path], comparisons: list[dict[str, Any]]
) -> None:
    """Persist gate/recovery, calibration, and method diagnostics for one analyzed split."""
    write_json(
        paths["data"] / "gates_and_recovery.json",
        {"comparisons": apply_holm(comparisons)},
    )
    write_json(paths["data"] / "calibration_summary.json", calibration_summary(paths))
    write_json(
        paths["data"] / "method_diagnostics.json",
        {"rows": method_diagnostics_summary(paths)},
    )


def _accumulate_diagnostics(entry: dict[str, Any], diagnostics: dict[str, Any]) -> None:
    """Fold one run's ``method_diagnostics`` into its (condition, method) entry.

    Generic over the diagnostic's name: a numeric value is summed across seeds, a
    list value has its length summed. This is what lets ``ssb_invalid_draws``
    (semantic-scale) and ``sc_mil_batch_diagnostics`` (SC-MIL) share one reader
    instead of each needing its own report path.
    """
    for name, value in diagnostics.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            entry[name] = entry.get(name, 0) + value
        elif isinstance(value, list):
            key = f"{name}_count"
            entry[key] = entry.get(key, 0) + len(value)


def method_diagnostics_summary(paths: dict[str, Path]) -> list[dict[str, Any]]:
    """Per-(condition, method) rollup of every ``method_diagnostics`` key any run recorded."""
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for condition, method, _seed_idx, result_dir in discover_result_dirs(
        paths["results"]
    ):
        record = read_run_record(result_dir, array_fields=())
        if record is None:
            continue
        diagnostics = record.get("method_diagnostics") or {}
        if not diagnostics:
            continue
        entry = rows.setdefault(
            (condition, method), {"condition": condition, "method": method, "seeds": 0}
        )
        entry["seeds"] += 1
        _accumulate_diagnostics(entry, diagnostics)
    return list(rows.values())


def _run_calibration(record: dict[str, Any]) -> dict[str, Any] | None:
    """Fit temperature on one run's validation logits; report raw vs. calibrated test NLL."""
    if "validation" not in record["splits"] or "test" not in record["splits"]:
        return None
    val, test = record["splits"]["validation"], record["splits"]["test"]
    val_logits = np.array(val["logits"])
    test_logits = np.array(test["logits"])
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
    for count, (condition, method, seed_idx, result_dir) in enumerate(
        discover_result_dirs(paths["results"]), start=1
    ):
        record = read_run_record(
            result_dir,
            splits=("validation", "test"),
            array_fields=("labels", "logits", "probabilities", "raw_probabilities"),
        )
        if record is not None:
            entry = _run_calibration(record)
            if entry is not None:
                assignment = record.get("assignment", "native")
                run_id = f"{record.get('benchmark', 'unknown')}:{assignment}:{condition}:{method}:seed={seed_idx}"
                summary[run_id] = entry
        if count % 25 == 0:
            logger.info("analyze: calibrated %d records", count)
    logger.info("analyze: calibrated %d records", len(summary))
    return summary
