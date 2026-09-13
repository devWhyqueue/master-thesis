"""Constants and config addressing for the concentrated-coverage experiment."""

from __future__ import annotations

from neighbours import DEEP_CELL, MAX_LEAKAGE, N_DRAWS, N_SPLITS, THRESHOLD_PP
from neighbours import allocation_dir, exp7_config
from neighbours.analyze import LowArm

__all__ = [
    "N_DRAWS",
    "N_SPLITS",
    "DEEP_CELL",
    "ALLOCATIONS",
    "MAX_LEAKAGE",
    "FOCAL_BASE_SEED",
    "THRESHOLD_PP",
    "FIT_SHARD_COUNT",
    "CONCENTRATED_ARM",
    "exp7_config",
    "allocation_dir",
]

# (patients G, patches per patient m); both broad allocations share one cell.
ALLOCATIONS: dict[str, tuple[int, int]] = {"concentrated": (20, 8), "random": (20, 8)}

FOCAL_BASE_SEED: int = 20260913

FIT_SHARD_COUNT: int = N_SPLITS * len(ALLOCATIONS)  # 6

CONCENTRATED_ARM = LowArm("concentrated", "within_concentration_residual", "b_c")
