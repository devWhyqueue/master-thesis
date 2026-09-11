"""Constants and config addressing for the site-coverage experiment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from breadth import N_DRAWS, N_SPLITS

__all__ = [
    "N_DRAWS",
    "N_SPLITS",
    "ALLOCATIONS",
    "N_SITES",
    "N_CORE_SITES",
    "PATIENTS_PER_SITE",
    "MIN_SITE_CLASSES",
    "BACKGROUND_CELL",
    "SITE_BASE_SEED",
    "THRESHOLD_PP",
    "FIT_SHARD_COUNT",
    "exp6_config",
    "allocation_dir",
]

# (patients G, patches per patient m) shared by all three allocations; site
# count (5 vs 10) is enforced by the allocation logic, not this table.
ALLOCATIONS: dict[str, tuple[int, int]] = {
    "deep": (5, 32),
    "broad5": (20, 8),
    "broad10": (20, 8),
}

N_SITES: int = 10
N_CORE_SITES: int = 5
PATIENTS_PER_SITE: int = 4
MIN_SITE_CLASSES: int = 10
BACKGROUND_CELL: tuple[int, int] = (10, 16)
SITE_BASE_SEED: int = 20260911
THRESHOLD_PP: float = 1.0

FIT_SHARD_COUNT: int = N_SPLITS * len(ALLOCATIONS)  # 9


def exp6_config(config: dict[str, Any]) -> dict[str, Any]:
    """View this config's stored exp-6 outputs as a standalone exp-6 config."""
    exp6_out = config.get("slurm", {}).get("exp6_outputs")
    if not exp6_out:
        raise ValueError("Config missing slurm.exp6_outputs")
    return {**config, "paths": {"outputs": str(exp6_out)}}


def allocation_dir(paths: dict[str, Path], allocation: str, draw_index: int) -> Path:
    """Return the result directory for one (allocation, draw) fit."""
    return paths["results"] / allocation / f"draw_{draw_index}"
