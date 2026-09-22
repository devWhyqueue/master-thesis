"""Shared run-record and split-path reading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from imbalance_benchmark.common import ensure_dirs, read_run_record, split_paths

from centre import N_SPLITS

__all__ = ["paths_by_split", "require_record"]


def paths_by_split(config: dict[str, Any]) -> dict[int, dict[str, Path]]:
    """Result directories for every split."""
    return {s: split_paths(ensure_dirs(config), s) for s in range(N_SPLITS)}


def require_record(
    result_dir: Path,
    splits: tuple[str, ...] | None = None,
    array_fields: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Read one stored run record, raising if it is missing."""
    rec = read_run_record(result_dir, splits=splits, array_fields=array_fields)
    if rec is None:
        raise RuntimeError(f"Missing run record at {result_dir}")
    return rec
