from __future__ import annotations

import argparse
import logging
from pathlib import Path

from scripts.common import compact_run_record, ensure_dirs, load_config
from scripts.analysis.results.ingest import discover_result_dirs

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse result-storage compaction arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--keep-tuning-arrays",
        action="store_true",
        help="Preserve dense sidecars for tuning runs instead of dropping them.",
    )
    parser.add_argument(
        "--keep-db",
        action="store_true",
        help="Do not remove SQLite files after compacting run records.",
    )
    return parser.parse_args()


def main() -> None:
    """Compact run records and remove stale large SQLite artifacts."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = parse_args()
    paths = ensure_dirs(load_config(args.config))
    report_count = _compact_report_runs(paths)
    tuning_count = _compact_tuning_runs(paths, args.keep_tuning_arrays)
    removed = [] if args.keep_db else _remove_database_files(paths["db"])
    logger.info(
        "Compacted %s report runs and %s tuning runs; removed %s DB files",
        report_count,
        tuning_count,
        len(removed),
    )


def _compact_report_runs(paths: dict[str, Path]) -> int:
    count = 0
    for result_dir, _benchmark, _method, _seed, _tuning_id in discover_result_dirs(
        paths
    ):
        count += int(compact_run_record(result_dir, keep_arrays=True))
    return count


def _compact_tuning_runs(paths: dict[str, Path], keep_arrays: bool) -> int:
    count = 0
    for result_dir, _benchmark, _method, _seed, tuning_id in discover_result_dirs(
        paths, include_tuning=True
    ):
        if tuning_id is None:
            continue
        count += int(compact_run_record(result_dir, keep_arrays=keep_arrays))
    return count


def _remove_database_files(db_path: Path) -> list[Path]:
    removed = []
    db_files = (
        db_path,
        db_path.with_name(f"{db_path.name}-wal"),
        db_path.with_name(f"{db_path.name}-shm"),
    )
    for path in db_files:
        if path.exists():
            path.unlink()
            removed.append(path)
    return removed


if __name__ == "__main__":
    main()
