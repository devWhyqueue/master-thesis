from __future__ import annotations

import argparse
import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.common import ensure_dirs, load_config, read_run_record
from scripts.analysis.results import (
    connect,
    discover_result_dirs,
    ingest_run_record,
    init_schema,
    replace_table,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse database build arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--include-tuning",
        action="store_true",
        help="Also ingest raw tuning runs; disabled by default to keep DB small.",
    )
    return parser.parse_args()


def _ingest_class_distribution(
    connection: sqlite3.Connection, paths: dict[str, Path]
) -> None:
    class_distribution = paths["tables"] / "class_distribution.csv"
    if class_distribution.exists():
        replace_table(
            connection,
            "dataset_class_distribution",
            pd.read_csv(class_distribution),
        )


def _ingest_split_distribution(
    connection: sqlite3.Connection, paths: dict[str, Path]
) -> None:
    for split_path in sorted(paths["tables"].glob("split_distribution_seed=*.csv")):
        frame = pd.read_csv(split_path)
        frame["seed"] = int(split_path.stem.split("=", 1)[1])
        replace_table(connection, "dataset_split_distribution", frame)
        return


def _ingest_dataset_stats(
    connection: sqlite3.Connection, paths: dict[str, Path]
) -> None:
    dataset_stats = paths["tables"] / "dataset_stats.json"
    if not dataset_stats.exists():
        return
    payload = json.loads(dataset_stats.read_text(encoding="utf-8"))
    replace_table(
        connection,
        "dataset_stats",
        pd.DataFrame([{"payload_json": json.dumps(payload, sort_keys=True)}]),
    )


def _ingest_progan_diagnostics(
    connection: sqlite3.Connection, paths: dict[str, Path]
) -> None:
    for diagnostics_path in sorted(
        paths["tables"].glob("progan_diagnostics_seed*.csv")
    ):
        frame = pd.read_csv(diagnostics_path)
        seed = int(diagnostics_path.stem.replace("progan_diagnostics_seed", ""))
        if "seed" not in frame.columns:
            frame.insert(0, "seed", seed)
        replace_table(connection, "progan_diagnostics", frame)
        return


def _ingest_wsi_profile(connection: sqlite3.Connection, paths: dict[str, Path]) -> None:
    for profile_path in sorted(paths["tables"].glob("wsi_bag_profile_seed=*.json")):
        payload = json.loads(profile_path.read_text(encoding="utf-8"))
        seed = int(profile_path.stem.split("=", 1)[1])
        replace_table(
            connection,
            "wsi_bag_profile",
            pd.DataFrame(
                [{"seed": seed, "payload_json": json.dumps(payload, sort_keys=True)}]
            ),
        )
        return


def ingest_reference_tables(
    connection: sqlite3.Connection, paths: dict[str, Path]
) -> None:
    """Load dataset and diagnostic reference artifacts into the database."""
    _ingest_class_distribution(connection, paths)
    _ingest_split_distribution(connection, paths)
    _ingest_dataset_stats(connection, paths)
    _ingest_progan_diagnostics(connection, paths)
    _ingest_wsi_profile(connection, paths)


def build_database(
    paths: dict[str, Path], include_tuning: bool = False
) -> dict[str, Any]:
    """Ingest all discovered run records and reference tables."""
    connection = connect(paths["db"])
    init_schema(connection)
    ingested = 0
    skipped = 0
    for result_dir, benchmark, method, seed, tuning_id in discover_result_dirs(
        paths, include_tuning=include_tuning
    ):
        if read_run_record(result_dir) is None:
            skipped += 1
            continue
        if ingest_run_record(
            connection, result_dir, benchmark, method, seed, tuning_id
        ):
            ingested += 1
    ingest_reference_tables(connection, paths)
    connection.close()
    return {"ingested": ingested, "skipped": skipped, "database": str(paths["db"])}


def main() -> None:
    """Build or refresh the consolidated experiment database."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = parse_args()
    paths = ensure_dirs(load_config(args.config))
    summary = build_database(paths, include_tuning=args.include_tuning)
    logger.info(
        "Built %s with %s runs (%s directories without payloads)",
        summary["database"],
        summary["ingested"],
        summary["skipped"],
    )


if __name__ == "__main__":
    main()
