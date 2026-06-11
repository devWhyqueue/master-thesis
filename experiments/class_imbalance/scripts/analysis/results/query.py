from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, cast

import pandas as pd

from scripts.analysis.results.core import SUMMARY_METRICS, read_table


def load_runs_frame(
    connection: sqlite3.Connection,
    benchmark: str,
    methods: list[str],
    seeds: list[int],
    *,
    report_only: bool = True,
) -> pd.DataFrame:
    """Return run metadata filtered to one report benchmark."""
    frame = read_table(connection, "runs")
    if frame.empty:
        return frame
    storage_benchmark = "patch_feature" if benchmark == "patch" else benchmark
    selected = cast(
        pd.DataFrame,
        frame[
            (frame["benchmark"] == storage_benchmark)
            & (frame["method"].isin(methods))
            & (frame["seed"].isin(seeds))
        ],
    )
    if report_only:
        selected = cast(
            pd.DataFrame,
            selected[selected["tuning_id"].isna() | (selected["tuning_id"] == "")],
        )
    return selected.reset_index(drop=True)


def load_eval_details(
    connection: sqlite3.Connection,
    benchmark: str,
    methods: list[str],
    seeds: list[int],
    split: str,
    *,
    report_only: bool = True,
) -> list[dict[str, Any]]:
    """Load per-run result payloads in the legacy jsonl detail shape."""
    runs = load_runs_frame(
        connection, benchmark, methods, seeds, report_only=report_only
    )
    if runs.empty:
        return []
    eval_results = read_table(connection, "eval_results")
    if eval_results.empty:
        return []
    merged = runs.merge(
        eval_results[eval_results["split"] == split],
        on="run_id",
        how="inner",
    )
    return [_detail_row(row, split) for row in merged.to_dict("records")]


def _detail_row(row: dict[str, Any], split: str) -> dict[str, Any]:
    metadata = (
        json.loads(row["method_metadata_json"])
        if row.get("method_metadata_json")
        else {}
    )
    extended = json.loads(row["extended_json"]) if row.get("extended_json") else {}
    result = {**extended}
    for metric in SUMMARY_METRICS:
        if metric in row and row[metric] is not None:
            result[metric] = row[metric]
    return {
        "method": row["method"],
        "method_metadata": metadata,
        "seed": int(row["seed"]),
        "split": split,
        "result": result,
    }


def _attach_arrays(
    payload: dict[str, Any], array_row: dict[str, Any]
) -> dict[str, Any]:
    for field, column in (
        ("labels", "labels_json"),
        ("preds", "preds_json"),
        ("probabilities", "probabilities_json"),
        ("class_names", "class_names_json"),
    ):
        raw = array_row.get(column)
        if isinstance(raw, str) and raw:
            payload[field] = json.loads(raw)
    return payload


def load_split_payload(
    connection: sqlite3.Connection,
    benchmark: str,
    method: str,
    seed: int,
    split: str,
    *,
    tuning_id: str | None = None,
) -> dict[str, Any] | None:
    """Reconstruct one split payload including arrays for calibration."""
    storage_benchmark = "patch_feature" if benchmark == "patch" else benchmark
    run_id = _lookup_run_id(
        connection, storage_benchmark, method, seed, tuning_id=tuning_id
    )
    if run_id is None:
        return None
    row = _split_result_row(connection, run_id, split)
    if row is None:
        return None
    payload = _payload_from_result_row(row)
    array_row = _split_array_row(connection, run_id, split)
    if array_row is None:
        return payload
    return _attach_arrays(payload, array_row)


def _split_result_row(
    connection: sqlite3.Connection, run_id: str, split: str
) -> dict[str, Any] | None:
    cursor = connection.execute(
        "SELECT * FROM eval_results WHERE run_id = ? AND split = ?",
        (run_id, split),
    )
    return _fetch_mapping(cursor)


def _split_array_row(
    connection: sqlite3.Connection, run_id: str, split: str
) -> dict[str, Any] | None:
    cursor = connection.execute(
        "SELECT * FROM eval_arrays WHERE run_id = ? AND split = ?",
        (run_id, split),
    )
    return _fetch_mapping(cursor)


def _fetch_mapping(cursor: sqlite3.Cursor) -> dict[str, Any] | None:
    row = cursor.fetchone()
    if row is None:
        return None
    columns = [description[0] for description in cursor.description]
    return dict(zip(columns, row, strict=True))


def _payload_from_result_row(row: dict[str, Any]) -> dict[str, Any]:
    payload = json.loads(row["extended_json"]) if row.get("extended_json") else {}
    for metric in SUMMARY_METRICS:
        if row.get(metric) is not None:
            payload[metric] = row[metric]
    return payload


def _lookup_run_id(
    connection: sqlite3.Connection,
    benchmark: str,
    method: str,
    seed: int,
    *,
    tuning_id: str | None,
) -> str | None:
    if tuning_id is None:
        cursor = connection.execute(
            """
            SELECT run_id FROM runs
            WHERE benchmark = ? AND method = ? AND seed = ?
              AND (tuning_id IS NULL OR tuning_id = '')
            LIMIT 1
            """,
            (benchmark, method, seed),
        )
    else:
        cursor = connection.execute(
            """
            SELECT run_id FROM runs
            WHERE benchmark = ? AND method = ? AND seed = ? AND tuning_id = ?
            LIMIT 1
            """,
            (benchmark, method, seed, tuning_id),
        )
    row = cursor.fetchone()
    if row is None:
        return None
    return str(row[0])


def load_class_distribution(
    connection: sqlite3.Connection, paths: dict[str, Path]
) -> pd.DataFrame:
    """Load class distribution from the database or legacy CSV."""
    frame = read_table(connection, "dataset_class_distribution")
    if not frame.empty:
        return frame
    csv_path = paths["tables"] / "class_distribution.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame()


def load_summary_by_seed(
    connection: sqlite3.Connection, benchmark: str, split: str | None = None
) -> pd.DataFrame:
    """Read per-seed summary rows for one benchmark."""
    return _filter_summary(read_table(connection, "summary_by_seed"), benchmark, split)


def load_summary(
    connection: sqlite3.Connection, benchmark: str, split: str | None = None
) -> pd.DataFrame:
    """Read aggregated summary rows for one benchmark."""
    return _filter_summary(read_table(connection, "summary"), benchmark, split)


def _filter_summary(
    frame: pd.DataFrame, benchmark: str, split: str | None
) -> pd.DataFrame:
    if frame.empty:
        return frame
    selected = cast(pd.DataFrame, frame[frame["benchmark"] == benchmark])
    if split is not None:
        selected = cast(pd.DataFrame, selected[selected["split"] == split])
    return selected.reset_index(drop=True)
