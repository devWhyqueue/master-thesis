from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.analysis.results.core import SUMMARY_METRICS, read_table, run_id_for_record


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
    selected = frame[
        (frame["benchmark"] == storage_benchmark)
        & (frame["method"].isin(methods))
        & (frame["seed"].isin(seeds))
    ]
    if report_only:
        selected = selected[
            selected["tuning_id"].isna() | (selected["tuning_id"] == "")
        ]
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


def _attach_arrays(payload: dict[str, Any], array_row: pd.Series) -> dict[str, Any]:
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
    run_id = run_id_for_record(storage_benchmark, method, seed, tuning_id)
    eval_results = read_table(connection, "eval_results")
    if eval_results.empty:
        return None
    rows = eval_results[
        (eval_results["run_id"] == run_id) & (eval_results["split"] == split)
    ]
    if rows.empty:
        return None
    row = rows.iloc[0].to_dict()
    payload = json.loads(row["extended_json"]) if row.get("extended_json") else {}
    for metric in SUMMARY_METRICS:
        if row.get(metric) is not None:
            payload[metric] = row[metric]
    eval_arrays = read_table(connection, "eval_arrays")
    if eval_arrays.empty:
        return payload
    array_rows = eval_arrays[
        (eval_arrays["run_id"] == run_id) & (eval_arrays["split"] == split)
    ]
    if array_rows.empty:
        return payload
    return _attach_arrays(payload, array_rows.iloc[0])


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
    selected = frame[frame["benchmark"] == benchmark]
    if split is not None:
        selected = selected[selected["split"] == split]
    return selected.reset_index(drop=True)
