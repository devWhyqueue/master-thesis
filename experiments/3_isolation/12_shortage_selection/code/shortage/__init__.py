"""Constants and config addressing for the shortage-selection experiment."""

from __future__ import annotations

from typing import Any

from neighbours import N_DRAWS, N_SPLITS, THRESHOLD_PP, exp7_config

from coverage_redundancy import exp5_config, exp10_config

__all__ = [
    "N_DRAWS",
    "N_SPLITS",
    "ALLOCATIONS",
    "MIN_RHO_SEL",
    "THRESHOLD_PP",
    "FIT_SHARD_COUNT",
    "exp5_config",
    "exp7_config",
    "exp10_config",
    "exp11_config",
]

# (patients G, patches per patient m); both arms share one cell.
ALLOCATIONS: dict[str, tuple[int, int]] = {"random": (5, 32), "dispersed": (5, 32)}

MIN_RHO_SEL: float = 0.5

FIT_SHARD_COUNT: int = N_SPLITS * len(ALLOCATIONS)  # 6


def exp11_config(config: dict[str, Any]) -> dict[str, Any]:
    """View this config's stored exp-11 outputs as a standalone exp-11 config."""
    exp11_out = config.get("slurm", {}).get("exp11_outputs")
    if not exp11_out:
        raise ValueError("Config missing slurm.exp11_outputs")
    return {**config, "paths": {"outputs": str(exp11_out)}}
