from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.io.sql import DatabaseError as PandasDatabaseError

SUMMARY_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "negative_log_likelihood",
    "brier_score",
    "expected_calibration_error",
)

ARRAY_FIELDS = ("labels", "preds", "probabilities")

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    result_dir TEXT NOT NULL,
    benchmark TEXT NOT NULL,
    method TEXT NOT NULL,
    seed INTEGER NOT NULL,
    tuning_id TEXT,
    tuning_params_json TEXT,
    smoke INTEGER NOT NULL DEFAULT 0,
    model_path TEXT,
    method_metadata_json TEXT,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS eval_results (
    run_id TEXT NOT NULL,
    split TEXT NOT NULL,
    accuracy REAL,
    balanced_accuracy REAL,
    macro_precision REAL,
    macro_recall REAL,
    macro_f1 REAL,
    negative_log_likelihood REAL,
    brier_score REAL,
    expected_calibration_error REAL,
    extended_json TEXT,
    PRIMARY KEY (run_id, split),
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS eval_arrays (
    run_id TEXT NOT NULL,
    split TEXT NOT NULL,
    labels_json TEXT,
    preds_json TEXT,
    probabilities_json TEXT,
    class_names_json TEXT,
    PRIMARY KEY (run_id, split),
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS run_diagnostics (
    run_id TEXT PRIMARY KEY,
    diagnostics_json TEXT,
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);
"""


def connect(db_file: Path) -> sqlite3.Connection:
    """Open the experiment database with WAL journaling enabled."""
    db_file.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_file)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA foreign_keys=ON")
    return connection


def init_schema(connection: sqlite3.Connection) -> None:
    """Create all result and report tables if they do not exist."""
    connection.executescript(_SCHEMA_SQL)
    connection.commit()


def replace_table(
    connection: sqlite3.Connection, table_name: str, frame: pd.DataFrame
) -> None:
    """Replace one table with the provided dataframe."""
    frame.to_sql(table_name, connection, if_exists="replace", index=False)


def read_table(connection: sqlite3.Connection, table_name: str) -> pd.DataFrame:
    """Read one table, returning an empty frame when it is missing."""
    try:
        return pd.read_sql(f'SELECT * FROM "{table_name}"', connection)
    except PandasDatabaseError:
        return pd.DataFrame()


def run_id_for_record(
    benchmark: str,
    method: str,
    seed: int,
    tuning_id: str | None = None,
) -> str:
    """Build a stable primary key for one run."""
    suffix = f":{tuning_id}" if tuning_id else ""
    return f"{benchmark}:{method}:seed={seed}{suffix}"


def write_json_table(
    connection: sqlite3.Connection, table_name: str, payload: dict[str, Any]
) -> None:
    """Store a single JSON document in a one-row table."""
    replace_table(
        connection,
        table_name,
        pd.DataFrame([{"payload_json": json.dumps(payload, sort_keys=True)}]),
    )
