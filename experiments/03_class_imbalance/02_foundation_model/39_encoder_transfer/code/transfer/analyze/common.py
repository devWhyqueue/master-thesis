"""Result-directory and run-record helpers shared by the analyze submodules."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from imbalance_benchmark.common import (
    N_PATIENT_SPLITS,
    ensure_dirs,
    read_run_record,
    split_paths,
)

from sites import allocation_dir

from transfer import MAIN_DRAWS

__all__ = ["N_DRAWS", "paths_by_split", "require_record", "fit_dirs"]

N_DRAWS = len(MAIN_DRAWS)


def paths_by_split(config: dict[str, Any]) -> dict[int, dict[str, Path]]:
    """Output-directory namespace for every locked patient split."""
    return {s: split_paths(ensure_dirs(config), s) for s in range(N_PATIENT_SPLITS)}


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


def fit_dirs(
    paths: dict[int, dict[str, Path]], encoder: str, arm: str
) -> list[tuple[int, int, Path]]:
    """(split, draw, result_dir) for every stored fit of one encoder's arm."""
    return [
        (s, d, allocation_dir(paths[s], f"{encoder}/{arm}", d))
        for s in range(N_PATIENT_SPLITS)
        for d in MAIN_DRAWS
    ]
