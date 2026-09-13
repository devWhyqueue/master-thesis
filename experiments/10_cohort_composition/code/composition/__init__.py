"""Constants and config addressing for the cohort-composition experiment."""

from __future__ import annotations

from neighbours import N_DRAWS, N_SPLITS, THRESHOLD_PP, allocation_dir, exp7_config

__all__ = [
    "N_DRAWS",
    "N_SPLITS",
    "ALLOCATIONS",
    "MIN_RHO_CON",
    "THRESHOLD_PP",
    "FIT_SHARD_COUNT",
    "exp7_config",
    "allocation_dir",
]

# (patients G, patches per patient m); all three allocations share one cell.
ALLOCATIONS: dict[str, tuple[int, int]] = {
    "clustered": (20, 8),
    "random": (20, 8),
    "dispersed": (20, 8),
}

MIN_RHO_CON: float = 0.5

FIT_SHARD_COUNT: int = N_SPLITS * len(ALLOCATIONS)  # 9
