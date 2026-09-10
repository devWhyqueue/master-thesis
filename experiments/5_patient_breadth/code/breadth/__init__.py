"""Constants, paths, and addressing for the patient-breadth experiment."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from decodability import (
    BOOTSTRAP_SEED,
    INPUT_DIM,
    N_REPLICATES,
    RELEVANCE_THRESHOLD_PP,
)
from imbalance_benchmark.common import ensure_dirs, split_paths

__all__ = [
    "BREADTH_LADDER",
    "DEPTH_LADDER",
    "FALLBACK_BREADTH_LADDER",
    "FALLBACK_DEPTH_LADDER",
    "GRID_CELLS",
    "N_DRAWS",
    "N_SPLITS",
    "FIT_SHARD_COUNT",
    "LAMBDAS",
    "TOLERANCE",
    "MAX_ITER",
    "BOOTSTRAP_SEED",
    "N_REPLICATES",
    "RELEVANCE_THRESHOLD_PP",
    "GAMMA_RELEVANCE_THRESHOLD",
    "INPUT_DIM",
    "TIE_TOLERANCE",
    "exp2_split_paths",
    "cell_label",
    "cell_dir",
    "draw_dir",
]

BREADTH_LADDER: tuple[int, ...] = (5, 10, 20)
DEPTH_LADDER: tuple[int, ...] = (8, 16, 32)
FALLBACK_BREADTH_LADDER: tuple[int, ...] = (4, 8, 16)
FALLBACK_DEPTH_LADDER: tuple[int, ...] = (4, 8, 16)

GRID_CELLS: tuple[tuple[int, int], ...] = tuple(
    (g, m) for g in BREADTH_LADDER for m in DEPTH_LADDER
)

N_DRAWS: int = 5
N_SPLITS: int = 3
FIT_SHARD_COUNT: int = N_SPLITS * len(GRID_CELLS)  # 27

LAMBDAS: tuple[float, ...] = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
TOLERANCE: float = 1e-8
MAX_ITER: int = 10000
TIE_TOLERANCE: float = 1e-10

GAMMA_RELEVANCE_THRESHOLD: float = 1.0 / math.log(2.0)  # ~1.4427 pp per log unit


def exp2_split_paths(config: dict[str, Any], split_index: int) -> dict[str, Path]:
    """Resolve exp-2 split paths for an exp-5 config."""
    exp2_out = config.get("slurm", {}).get("exp2_outputs")
    if not exp2_out:
        raise ValueError("Config missing slurm.exp2_outputs")
    base = ensure_dirs({"paths": {"outputs": str(exp2_out)}})
    return split_paths(base, split_index)


def cell_label(g: int, m: int) -> str:
    """Format grid cell identifier string, e.g. G20_m8."""
    return f"G{g}_m{m}"


def cell_dir(paths: dict[str, Path], g: int, m: int) -> Path:
    """Return result directory for one (G, m) grid cell."""
    return paths["results"] / cell_label(g, m)


def draw_dir(paths: dict[str, Path], g: int, m: int, draw_index: int) -> Path:
    """Return result directory for one (G, m, draw) run."""
    return cell_dir(paths, g, m) / f"draw_{draw_index}"

