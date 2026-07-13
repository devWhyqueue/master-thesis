from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from imbalance_benchmark.common import read_run_record

__all__ = [
    "load_runs_frame",
    "load_eval_details",
    "load_classwise",
    "load_split_payload",
    "load_test_identity",
    "load_seed_predictions",
]


def load_runs_frame(conn: sqlite3.Connection) -> pd.DataFrame:
    """Load the ``runs`` table."""
    return pd.read_sql_query("SELECT * FROM runs", conn)


def load_eval_details(conn: sqlite3.Connection) -> pd.DataFrame:
    """Load ``runs`` joined with ``eval_results``, one row per run x split."""
    return pd.read_sql_query(
        "SELECT r.*, e.split, e.accuracy, e.balanced_accuracy, e.macro_precision, "
        "e.macro_recall, e.macro_f1, e.macro_nll, e.negative_log_likelihood, "
        "e.brier_score, e.expected_calibration_error, e.quadratic_weighted_kappa, "
        "e.ordinal_mean_absolute_error, e.extended_json "
        "FROM runs r JOIN eval_results e ON r.run_id = e.run_id",
        conn,
    )


def load_classwise(conn: sqlite3.Connection) -> pd.DataFrame:
    """Load ``eval_classwise`` joined with ``runs`` identity columns."""
    return pd.read_sql_query(
        "SELECT r.run_id, r.benchmark, r.condition, r.method, r.seed_index, c.split, "
        "c.class_name, c.tier, c.precision, c.recall, c.f1, c.support, c.nll, c.brier "
        "FROM eval_classwise c JOIN runs r ON r.run_id = c.run_id",
        conn,
    )


def load_split_payload(result_dir: Path, split: str) -> dict[str, Any] | None:
    """Load one run's evaluated split (with labels/preds/probabilities/logits reattached)."""
    record = read_run_record(result_dir)
    if record is None:
        return None
    return record.get("splits", {}).get(split)


def load_test_identity(
    manifest_path: Path, is_mil: bool, split_name: str = "test"
) -> pd.DataFrame:
    """Rebuild the case_id/slide_id vector for a locked split in prediction-array order.

    Confirmation's ``DataLoader`` never shuffles, so this reproduces the exact
    row order ``ImbalanceDataset``/``BagFeatureDataset`` iterate in (one row
    per patch, or one row per slide via ``groupby("slide_id").first()``),
    letting every run's stored ``preds``/``labels``/``probabilities`` arrays be
    joined back to patient/slide identity by position without persisting
    patient IDs inside every run record.
    """
    df = pd.read_csv(manifest_path)
    df = cast(pd.DataFrame, df[df["split"] == split_name]).reset_index(drop=True)
    if is_mil:
        df = df.groupby("slide_id").first().reset_index()
    return cast(pd.DataFrame, df[["case_id", "slide_id", "cancer_type"]]).reset_index(
        drop=True
    )


def load_seed_predictions(
    paths: dict[str, Path], condition: str, method: str, assignment: str = "native"
) -> dict[str, Any] | None:
    """Stack one method's confirmed test-split predictions across its confirmation seeds."""
    assigned_dir = paths["results"] / f"assignment={assignment}" / condition / method
    method_dir = (
        assigned_dir if assigned_dir.exists() else paths["results"] / condition / method
    )
    if not method_dir.exists():
        return None
    seed_dirs = sorted(
        method_dir.glob("seed=*"), key=lambda p: int(p.name.split("=")[1])
    )
    records = [read_run_record(d) for d in seed_dirs]
    records = [r for r in records if r is not None and "test" in r.get("splits", {})]
    if not records:
        return None
    test_splits = [r["splits"]["test"] for r in records]
    return {
        "class_names": records[0].get("class_names", []),
        "labels": np.array(test_splits[0]["labels"]),
        "preds": np.stack([np.array(s["preds"]) for s in test_splits]),
        "probs": np.stack([np.array(s["probabilities"]) for s in test_splits]),
        "logits": np.stack([np.array(s["logits"]) for s in test_splits]),
    }
