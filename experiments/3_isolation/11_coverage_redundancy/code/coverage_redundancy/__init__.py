"""Constants and config addressing for the coverage-redundancy experiment."""

from __future__ import annotations

from typing import Any

from breadth import GRID_CELLS, N_REPLICATES
from neighbours import N_DRAWS, N_SPLITS, THRESHOLD_PP
from redundancy import exp5_config
from redundancy.surfaces import LN2
from sites import exp6_config

__all__ = [
    "N_SPLITS",
    "N_DRAWS",
    "GRID_CELLS",
    "N_REPLICATES",
    "THRESHOLD_PP",
    "LN2",
    "COMPOSITION_ALLOCATIONS",
    "exp5_config",
    "exp6_config",
    "exp10_config",
]

COMPOSITION_ALLOCATIONS: tuple[str, ...] = ("clustered", "random", "dispersed")


def exp10_config(config: dict[str, Any]) -> dict[str, Any]:
    """View this config's stored exp-10 outputs as a standalone exp-10 config."""
    exp10_out = config.get("slurm", {}).get("exp10_outputs")
    if not exp10_out:
        raise ValueError("Config missing slurm.exp10_outputs")
    return {**config, "paths": {"outputs": str(exp10_out)}}
