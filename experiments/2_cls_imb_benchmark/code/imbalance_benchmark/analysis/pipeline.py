from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

from imbalance_benchmark.analysis.calibration import (
    apply_target_prior_correction,
    apply_temperature,
    balanced_decision_logits,
    estimate_prior,
    fit_temperature,
)
from imbalance_benchmark.analysis.db import discover_result_dirs, ingest_run
from imbalance_benchmark.analysis.metrics import assign_tiers, negative_log_likelihood
from imbalance_benchmark.common import read_run_record

__all__ = ["ingest_all_runs", "calibration_summary"]


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
        class_names = record.get("class_names", [])
        allocated = (
            freeze.get("conditions", {}).get(condition, {}).get("allocated_counts", {})
        )
        tiers = (
            assign_tiers(class_names, allocated) if class_names and allocated else {}
        )
        run_id = (
            f"{record.get('benchmark', 'unknown')}:{condition}:{method}:seed={seed_idx}"
        )
        ingest_run(conn, run_id, result_dir, condition, method, seed_idx, record, tiers)


def _run_calibration(record: dict[str, Any], method: str) -> dict[str, Any] | None:
    """Fit temperature on one run's validation logits; report raw vs. calibrated test NLL."""
    if "validation" not in record["splits"] or "test" not in record["splits"]:
        return None
    val, test = record["splits"]["validation"], record["splits"]["test"]
    class_names = record.get("class_names", [])
    val_logits, test_logits = np.array(val["logits"]), np.array(test["logits"])
    tau = float(record.get("tuning_params", {}).get("parameter", 1.0))
    if method in ("post_hoc_logit_adjustment", "logit_adjustment") and class_names:
        pi_train = estimate_prior(np.array(val["labels"]), len(class_names))
        val_logits = balanced_decision_logits(val_logits, method, tau, pi_train)
        test_logits = apply_target_prior_correction(
            test_logits, method, tau, pi_train, pi_train
        )
    fit = fit_temperature(val_logits, np.array(val["labels"]))
    calibrated_probs = apply_temperature(test_logits, fit.temperature)
    return {
        "temperature": fit.temperature,
        "raw_test_nll": test["negative_log_likelihood"],
        "temperature_scaled_test_nll": negative_log_likelihood(
            np.array(test["labels"]), calibrated_probs
        ),
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
        entry = _run_calibration(record, method)
        if entry is not None:
            run_id = f"{record.get('benchmark', 'unknown')}:{condition}:{method}:seed={seed_idx}"
            summary[run_id] = entry
    return summary
