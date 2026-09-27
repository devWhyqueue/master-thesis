"""Shard keys and result directories for the exp-41 analyze stage."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from imbalance_benchmark.common import ensure_dirs, split_paths

from sites import allocation_dir

from centre import N_DRAWS, N_SPLITS

from mitigation import method_label

__all__ = ["Shard", "shard_keys", "result_paths", "run_dir", "fixed_dirs"]

Shard = tuple[int, int]


def shard_keys() -> list[Shard]:
    """Every (split, draw) key of the design, in fit/pooling order."""
    return [(s, d) for s in range(N_SPLITS) for d in range(N_DRAWS)]


def result_paths(config: dict[str, Any]) -> dict[int, dict[str, Path]]:
    """Result-directory roots of every split, matching ``mitigation.data.load_shard``."""
    return {s: split_paths(ensure_dirs(config), s) for s in range(N_SPLITS)}


def run_dir(
    paths: dict[str, Path], arm: str, method: str, param: float | None, d: int
) -> Path:
    """One (arm, method, param, draw) run's result directory, matching ``mitigation.data.run_dir``."""
    return allocation_dir(paths, f"{arm}_{method_label(method, param)}", d)


def fixed_dirs(
    paths: dict[int, dict[str, Path]], arm: str, method: str, param: float | None
) -> dict[Shard, Path]:
    """The same (arm, method, param) run's directory across every shard."""
    return {(s, d): run_dir(paths[s], arm, method, param, d) for s, d in shard_keys()}
