from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

__all__ = ["connect_db", "init_schema", "discover_result_dirs", "ingest_run"]

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY, result_dir TEXT NOT NULL, benchmark TEXT NOT NULL,
    condition TEXT NOT NULL, method TEXT NOT NULL, seed_index INTEGER NOT NULL,
    seed INTEGER, class_names_json TEXT, tuning_params_json TEXT, cost_json TEXT,
    smoke INTEGER NOT NULL DEFAULT 0, created_at TEXT
);
CREATE TABLE IF NOT EXISTS eval_results (
    run_id TEXT NOT NULL, split TEXT NOT NULL, accuracy REAL,
    balanced_accuracy REAL, macro_precision REAL, macro_recall REAL,
    macro_f1 REAL, macro_nll REAL, negative_log_likelihood REAL, brier_score REAL,
    expected_calibration_error REAL, quadratic_weighted_kappa REAL,
    ordinal_mean_absolute_error REAL, extended_json TEXT,
    PRIMARY KEY (run_id, split), FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS eval_classwise (
    run_id TEXT NOT NULL, split TEXT NOT NULL, class_name TEXT NOT NULL, tier TEXT,
    precision REAL, recall REAL, f1 REAL, support INTEGER, nll REAL, brier REAL,
    PRIMARY KEY (run_id, split, class_name),
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);
"""

_EVAL_SCALAR_FIELDS = (
    "accuracy",
    "balanced_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "macro_nll",
    "negative_log_likelihood",
    "brier_score",
    "expected_calibration_error",
    "quadratic_weighted_kappa",
    "ordinal_mean_absolute_error",
)
_CLASSWISE_ARRAY_FIELDS = (
    "precision_per_class",
    "recall_per_class",
    "f1_per_class",
    "support_per_class",
    "nll_per_class",
    "brier_per_class",
)


def connect_db(db_file: Path) -> sqlite3.Connection:
    """Connect to SQLite database and apply performance PRAGMAs."""
    db_file.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_file)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    """Initialize the schema and migrate result databases created by earlier versions."""
    conn.executescript(_SCHEMA_SQL)
    columns = {
        row[1] for row in conn.execute("PRAGMA table_info(eval_results)").fetchall()
    }
    for name in ("quadratic_weighted_kappa", "ordinal_mean_absolute_error"):
        if name not in columns:
            conn.execute(f"ALTER TABLE eval_results ADD COLUMN {name} REAL")
    conn.commit()


def discover_result_dirs(results_root: Path) -> list[tuple[str, str, int, Path]]:
    """Walk ``results/<condition>/<method>/seed=<i>/`` generically for run.json dirs.

    Replaces the previous hard-coded ``["balanced","moderate"] x ["ce",
    "weighted_ce"] x range(5)`` enumeration with directory discovery, so every
    condition/method/seed actually confirmed is ingested.
    """
    found = []
    if not results_root.exists():
        return found
    for run_file in sorted(results_root.rglob("run.json")):
        with run_file.open(encoding="utf-8") as handle:
            record = json.load(handle)
        seed_dir = run_file.parent
        found.append(
            (
                str(record["condition"]),
                str(record["method"]),
                int(seed_dir.name.split("=", 1)[1]),
                seed_dir,
            )
        )
    return found


def _ingest_run_meta(
    conn: sqlite3.Connection,
    run_id: str,
    result_dir: Path,
    condition: str,
    method: str,
    seed_index: int,
    record: dict[str, Any],
) -> None:
    """Helper to insert runs metadata."""
    conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
    conn.execute(
        "INSERT INTO runs (run_id, result_dir, benchmark, condition, method, seed_index, "
        "seed, class_names_json, tuning_params_json, cost_json, smoke, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))",
        (
            run_id,
            str(result_dir),
            record.get("benchmark", "unknown"),
            condition,
            method,
            seed_index,
            record.get("seed"),
            json.dumps(record.get("class_names", []), sort_keys=False),
            json.dumps(record.get("tuning_params", {}), sort_keys=True),
            json.dumps(record.get("cost", {}), sort_keys=True),
            int(bool(record.get("smoke", False))),
        ),
    )


def _ingest_classwise(
    conn: sqlite3.Connection,
    run_id: str,
    split_name: str,
    payload: dict[str, Any],
    class_names: list[str],
    tiers: dict[str, str],
) -> None:
    """Insert one split's per-class precision/recall/f1/support/NLL/Brier rows."""
    conn.execute(
        "DELETE FROM eval_classwise WHERE run_id = ? AND split = ?",
        (run_id, split_name),
    )
    if not class_names or "precision_per_class" not in payload:
        return
    arrays = {field: payload.get(field, []) for field in _CLASSWISE_ARRAY_FIELDS}
    for i, name in enumerate(class_names):
        conn.execute(
            "INSERT INTO eval_classwise (run_id, split, class_name, tier, precision, "
            "recall, f1, support, nll, brier) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                split_name,
                name,
                tiers.get(name),
                arrays["precision_per_class"][i]
                if i < len(arrays["precision_per_class"])
                else None,
                arrays["recall_per_class"][i]
                if i < len(arrays["recall_per_class"])
                else None,
                arrays["f1_per_class"][i] if i < len(arrays["f1_per_class"]) else None,
                arrays["support_per_class"][i]
                if i < len(arrays["support_per_class"])
                else None,
                arrays["nll_per_class"][i]
                if i < len(arrays["nll_per_class"])
                else None,
                arrays["brier_per_class"][i]
                if i < len(arrays["brier_per_class"])
                else None,
            ),
        )


def _ingest_run_splits(
    conn: sqlite3.Connection,
    run_id: str,
    splits: dict[str, Any],
    class_names: list[str],
    tiers: dict[str, str],
) -> None:
    """Helper to insert split-level scalar metrics and classwise rows."""
    for split_name, payload in splits.items():
        if not isinstance(payload, dict):
            continue
        ext = {
            k: v
            for k, v in payload.items()
            if k not in _EVAL_SCALAR_FIELDS
            and k not in _CLASSWISE_ARRAY_FIELDS
            and k not in ("labels", "preds", "probabilities", "logits")
        }
        conn.execute(
            "DELETE FROM eval_results WHERE run_id = ? AND split = ?",
            (run_id, split_name),
        )
        conn.execute(
            "INSERT INTO eval_results (run_id, split, accuracy, balanced_accuracy, "
            "macro_precision, macro_recall, macro_f1, macro_nll, negative_log_likelihood, "
            "brier_score, expected_calibration_error, quadratic_weighted_kappa, "
            "ordinal_mean_absolute_error, extended_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                split_name,
                *(payload.get(field) for field in _EVAL_SCALAR_FIELDS),
                json.dumps(ext, sort_keys=True),
            ),
        )
        _ingest_classwise(conn, run_id, split_name, payload, class_names, tiers)


def ingest_run(
    conn: sqlite3.Connection,
    run_id: str,
    result_dir: Path,
    condition: str,
    method: str,
    seed_index: int,
    record: dict[str, Any],
    tiers: dict[str, str] | None = None,
) -> None:
    """Ingest a single run record and its evaluation splits into the SQLite database."""
    class_names = record.get("class_names", [])
    _ingest_run_meta(conn, run_id, result_dir, condition, method, seed_index, record)
    _ingest_run_splits(conn, run_id, record.get("splits", {}), class_names, tiers or {})
    conn.commit()
