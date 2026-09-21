"""Constants and config addressing for the patient-coverage experiment."""

from __future__ import annotations

from typing import Any

from breadth import N_DRAWS, N_SPLITS

from sites import allocation_dir, exp6_config

__all__ = [
    "N_DRAWS",
    "N_SPLITS",
    "DEEP_CELL",
    "ALLOCATIONS",
    "N_ROUNDS",
    "MAX_LEAKAGE",
    "NEIGHBOUR_BASE_SEED",
    "THRESHOLD_PP",
    "FIT_SHARD_COUNT",
    "exp6_config",
    "exp7_config",
    "allocation_dir",
]

DEEP_CELL: tuple[int, int] = (5, 32)
# (patients G, patches per patient m); both broad allocations share one cell.
ALLOCATIONS: dict[str, tuple[int, int]] = {"neighbours": (20, 8), "random": (20, 8)}

N_ROUNDS: int = 3
MAX_LEAKAGE: float = 0.5
NEIGHBOUR_BASE_SEED: int = 20260912
THRESHOLD_PP: float = 1.0

FIT_SHARD_COUNT: int = N_SPLITS * len(ALLOCATIONS)  # 6


def exp7_config(config: dict[str, Any]) -> dict[str, Any]:
    """View this config's stored exp-7 outputs as a standalone exp-7 config."""
    exp7_out = config.get("slurm", {}).get("exp7_outputs")
    if not exp7_out:
        raise ValueError("Config missing slurm.exp7_outputs")
    return {**config, "paths": {"outputs": str(exp7_out)}}
