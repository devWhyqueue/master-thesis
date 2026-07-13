from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "connect_db",
    "init_schema",
    "ingest_run",
    "compute_ece",
    "compute_brier_score",
    "run_bootstrap_preflight",
    "run_paired_permutation_test",
    "holm_adjust_pvalues",
]

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY, result_dir TEXT NOT NULL, benchmark TEXT NOT NULL,
    method TEXT NOT NULL, seed INTEGER NOT NULL, tuning_id TEXT,
    tuning_params_json TEXT, smoke INTEGER NOT NULL DEFAULT 0,
    model_path TEXT, method_metadata_json TEXT, created_at TEXT
);
CREATE TABLE IF NOT EXISTS eval_results (
    run_id TEXT NOT NULL, split TEXT NOT NULL, accuracy REAL,
    balanced_accuracy REAL, macro_precision REAL, macro_recall REAL,
    macro_f1 REAL, negative_log_likelihood REAL, brier_score REAL,
    expected_calibration_error REAL, extended_json TEXT,
    PRIMARY KEY (run_id, split), FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);
"""


def connect_db(db_file: Path) -> sqlite3.Connection:
    """Connect to SQLite database and apply performance PRAGMAs."""
    db_file.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_file)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    """Initialize SQL tables schema."""
    conn.executescript(_SCHEMA_SQL)
    conn.commit()


def _ingest_run_meta(
    conn: sqlite3.Connection, run_id: str, result_dir: Path, record: dict[str, Any]
) -> None:
    """Helper to insert runs metadata."""
    conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
    conn.execute(
        "INSERT INTO runs (run_id, result_dir, benchmark, method, seed, tuning_id, tuning_params_json, smoke, model_path, method_metadata_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))",
        (
            run_id,
            str(result_dir),
            record.get("benchmark", "unknown"),
            record.get("method", "unknown"),
            int(record.get("seed", 0)),
            record.get("tuning_id"),
            json.dumps(record.get("tuning_params", {}), sort_keys=True),
            int(bool(record.get("smoke", False))),
            record.get("model_path"),
            json.dumps(record.get("method_metadata"), sort_keys=True)
            if record.get("method_metadata")
            else None,
        ),
    )


def _ingest_run_splits(
    conn: sqlite3.Connection, run_id: str, splits: dict[str, Any]
) -> None:
    """Helper to insert split results."""
    for split_name, payload in splits.items():
        if not isinstance(payload, dict):
            continue
        ext = {
            k: v
            for k, v in payload.items()
            if k
            not in (
                "accuracy",
                "balanced_accuracy",
                "macro_precision",
                "macro_recall",
                "macro_f1",
                "negative_log_likelihood",
                "brier_score",
                "expected_calibration_error",
            )
        }
        conn.execute(
            "DELETE FROM eval_results WHERE run_id = ? AND split = ?",
            (run_id, split_name),
        )
        conn.execute(
            "INSERT INTO eval_results (run_id, split, accuracy, balanced_accuracy, macro_precision, macro_recall, macro_f1, negative_log_likelihood, brier_score, expected_calibration_error, extended_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                split_name,
                payload.get("accuracy"),
                payload.get("balanced_accuracy"),
                payload.get("macro_precision"),
                payload.get("macro_recall"),
                payload.get("macro_f1"),
                payload.get("negative_log_likelihood"),
                payload.get("brier_score"),
                payload.get("expected_calibration_error"),
                json.dumps(ext, sort_keys=True),
            ),
        )


def ingest_run(
    conn: sqlite3.Connection, run_id: str, result_dir: Path, record: dict[str, Any]
) -> None:
    """Ingest a single run record and its evaluation splits into the SQLite database."""
    _ingest_run_meta(conn, run_id, result_dir, record)
    _ingest_run_splits(conn, run_id, record.get("splits", {}))
    conn.commit()


def compute_ece(probs: np.ndarray, targets: np.ndarray, n_bins: int = 15) -> float:
    """Compute Expected Calibration Error (ECE)."""
    conf = np.max(probs, axis=1)
    accs = np.argmax(probs, axis=1) == targets
    ece = 0.0
    bins = np.linspace(0, 1, n_bins + 1)
    for i in range(n_bins):
        in_bin = (conf > bins[i]) & (conf <= bins[i + 1])
        if np.mean(in_bin) > 0:
            ece += np.mean(in_bin) * np.abs(
                np.mean(conf[in_bin]) - np.mean(accs[in_bin])
            )
    return float(ece)


def compute_brier_score(
    probs: np.ndarray, targets: np.ndarray, n_classes: int
) -> float:
    """Compute multiclass Brier score."""
    return float(np.mean(np.sum((probs - np.eye(n_classes)[targets]) ** 2, axis=1)))


def run_bootstrap_preflight(df_test: pd.DataFrame) -> dict[str, Any]:
    """Verify bootstrap constraints (Kish effective count and patient skewness)."""
    w = (
        df_test.groupby(["case_id", "cancer_type"])
        .size()
        .unstack(fill_value=0)
        .sum(axis=1)
        .to_numpy()
    )
    sum_w, sum_w2 = w.sum(), (w**2).sum()
    kish = float((sum_w**2) / sum_w2) if sum_w2 > 0 else 0.0
    max_w = float(w.max() / sum_w) if sum_w > 0 else 0.0
    return {
        "kish_effective_count": kish,
        "max_patient_weight_fraction": max_w,
        "is_descriptive_only": kish < 5.0 or max_w > 0.5,
    }


def run_paired_permutation_test(
    metrics_a: np.ndarray, metrics_b: np.ndarray, n_permutations: int = 100000
) -> float:
    """Run paired patient-block permutation test and return p-value."""
    diffs = metrics_a - metrics_b
    obs = np.mean(diffs)
    flips = np.random.default_rng(42).choice([-1, 1], size=(n_permutations, len(diffs)))
    extreme = np.sum(np.abs(np.mean(diffs * flips, axis=1)) >= np.abs(obs))
    return float((extreme + 1) / (n_permutations + 1))


def holm_adjust_pvalues(pvalues: list[float]) -> list[float]:
    """Apply Holm-Bonferroni correction to p-values."""
    m = len(pvalues)
    if m == 0:
        return []
    idx = np.argsort(pvalues)
    adj = [0.0] * m
    prev = 0.0
    for i, s_idx in enumerate(idx):
        prev = max(prev, min(1.0, pvalues[s_idx] * (m - i)))
        adj[s_idx] = prev
    return adj
