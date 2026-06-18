from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from scripts.common import read_run_record
from scripts.analysis.results.core import (
    ARRAY_FIELDS,
    SUMMARY_METRICS,
    run_id_for_record,
)


def ingest_run_record(
    connection: sqlite3.Connection,
    result_dir: Path,
    benchmark: str,
    method: str,
    seed: int,
    tuning_id: str | None = None,
) -> bool:
    """Insert or replace one run and its split payloads."""
    record = read_run_record(result_dir)
    if record is None:
        return False
    resolved_benchmark = _record_value(record, "benchmark", benchmark)
    resolved_method = _record_value(record, "method", method)
    run_id = run_id_for_record(
        resolved_benchmark,
        resolved_method,
        int(record.get("seed", seed)),
        record.get("tuning_id") or tuning_id,
    )
    _insert_run_row(
        connection, run_id, result_dir, record, benchmark, method, seed, tuning_id
    )
    splits = record.get("splits", {})
    if not isinstance(splits, dict):
        raise ValueError(f"Run record splits must be a mapping: {result_dir}")
    for split, payload in splits.items():
        if isinstance(payload, dict):
            _ingest_split(connection, run_id, str(split), payload)
            _delete_array_row(connection, run_id, str(split))
    _insert_diagnostics(connection, run_id, record.get("diagnostics"))
    connection.commit()
    return True


def _insert_run_row(
    connection: sqlite3.Connection,
    run_id: str,
    result_dir: Path,
    record: dict[str, Any],
    benchmark: str,
    method: str,
    seed: int,
    tuning_id: str | None,
) -> None:
    resolved_benchmark = _record_value(record, "benchmark", benchmark)
    resolved_method = _record_value(record, "method", method)
    connection.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
    connection.execute(
        """
        INSERT INTO runs (
            run_id, result_dir, benchmark, method, seed, tuning_id,
            tuning_params_json, smoke, model_path, method_metadata_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
        """,
        (
            run_id,
            str(result_dir),
            resolved_benchmark,
            resolved_method,
            int(record.get("seed", seed)),
            record.get("tuning_id") or tuning_id,
            json.dumps(record.get("tuning_params", {}), sort_keys=True),
            int(bool(record.get("smoke", False))),
            record.get("model_path"),
            json.dumps(record.get("method_metadata"), sort_keys=True)
            if record.get("method_metadata") is not None
            else None,
        ),
    )


def _record_value(record: dict[str, Any], key: str, fallback: str) -> str:
    value = str(record.get(key) or "")
    return fallback if value in {"", "unknown"} else value


def _insert_diagnostics(
    connection: sqlite3.Connection, run_id: str, diagnostics: object | None
) -> None:
    connection.execute("DELETE FROM run_diagnostics WHERE run_id = ?", (run_id,))
    if diagnostics is not None:
        connection.execute(
            "INSERT INTO run_diagnostics (run_id, diagnostics_json) VALUES (?, ?)",
            (run_id, json.dumps(diagnostics, sort_keys=True)),
        )


def _ingest_split(
    connection: sqlite3.Connection,
    run_id: str,
    split: str,
    payload: dict[str, Any],
) -> None:
    extended = {
        key: value
        for key, value in payload.items()
        if key not in SUMMARY_METRICS and key not in ARRAY_FIELDS
    }
    metric_values = [
        run_id,
        split,
        *(payload.get(metric) for metric in SUMMARY_METRICS),
        json.dumps(extended, sort_keys=True),
    ]
    columns = ", ".join(["run_id", "split", *SUMMARY_METRICS, "extended_json"])
    placeholders = ", ".join("?" for _ in metric_values)
    connection.execute(
        "DELETE FROM eval_results WHERE run_id = ? AND split = ?",
        (run_id, split),
    )
    connection.execute(
        f"INSERT INTO eval_results ({columns}) VALUES ({placeholders})",
        metric_values,
    )


def _delete_array_row(connection: sqlite3.Connection, run_id: str, split: str) -> None:
    connection.execute(
        "DELETE FROM eval_arrays WHERE run_id = ? AND split = ?",
        (run_id, split),
    )


def discover_result_dirs(
    paths: dict[str, Path],
    *,
    include_tuning: bool = False,
) -> list[tuple[Path, str, str, int, str | None]]:
    """Enumerate all per-run directories that may contain result payloads."""
    discovered: list[tuple[Path, str, str, int, str | None]] = []
    patch_feature_root = paths["results"] / "patch_feature"
    if patch_feature_root.exists():
        discovered.extend(_discover_method_dirs(patch_feature_root, "patch_feature"))
    if paths["patch_results"].exists():
        discovered.extend(_discover_method_dirs(paths["patch_results"], "patch_image"))
    if paths["wsi_results"].exists():
        discovered.extend(_discover_method_dirs(paths["wsi_results"], "wsi_bag"))
    if include_tuning:
        discovered.extend(_discover_tuning_dirs(paths["root"] / "outputs" / "tuning"))
    return discovered


def _discover_tuning_dirs(
    tuning_root: Path,
) -> list[tuple[Path, str, str, int, str | None]]:
    if not tuning_root.exists():
        return []
    discovered: list[tuple[Path, str, str, int, str | None]] = []
    for benchmark_dir in tuning_root.iterdir():
        if not benchmark_dir.is_dir():
            continue
        for method_dir in benchmark_dir.iterdir():
            if not method_dir.is_dir():
                continue
            for variant_dir in method_dir.iterdir():
                if not variant_dir.is_dir():
                    continue
                for seed_dir in variant_dir.iterdir():
                    if seed_dir.is_dir() and seed_dir.name.startswith("seed="):
                        seed = int(seed_dir.name.split("=", 1)[1])
                        discovered.append(
                            (
                                seed_dir,
                                benchmark_dir.name,
                                method_dir.name,
                                seed,
                                variant_dir.name,
                            )
                        )
    return discovered


def _discover_method_dirs(
    root: Path, benchmark: str
) -> list[tuple[Path, str, str, int, str | None]]:
    rows: list[tuple[Path, str, str, int, str | None]] = []
    for method_dir in root.iterdir():
        if not method_dir.is_dir():
            continue
        for seed_dir in method_dir.iterdir():
            if seed_dir.is_dir() and seed_dir.name.startswith("seed="):
                seed = int(seed_dir.name.split("=", 1)[1])
                rows.append((seed_dir, benchmark, method_dir.name, seed, None))
    return rows
