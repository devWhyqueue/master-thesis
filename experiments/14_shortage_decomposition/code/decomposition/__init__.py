"""Constants and config addressing for the shortage-decomposition experiment."""

from __future__ import annotations

from typing import Any

from breadth import N_REPLICATES, N_SPLITS

from neighbours import THRESHOLD_PP

__all__ = [
    "N_SPLITS",
    "N_REPLICATES",
    "THRESHOLD_PP",
    "N_DRAWS_DEFAULT",
    "DRAW_COUNTS",
    "N_SIM",
    "N_STUDY_REPLICATES",
    "HALFWIDTH_TOL_PP",
    "DISPERSION_SENSITIVITY",
    "SCENARIO_C",
    "SCENARIO_S",
    "SCENARIO_P",
    "SPREAD_R_FACTOR",
    "SPREAD_OMEGA_MIN",
    "SEPARATION_MAX_ABS_CORR",
    "OVERLAP_R_FACTOR",
    "OVERLAP_OMEGA_TOL",
    "OVERLAP_MIN_SHARE",
    "TRANSFER_MIN_RATIO",
    "exp5_config",
    "exp12_config",
]

# Census draw count before a precision.json exists (report Sec. "cohorts").
N_DRAWS_DEFAULT: int = 20

# Precision simulation (report App. "Precision simulation").
DRAW_COUNTS: tuple[int, ...] = (20, 40, 60)
N_SIM: int = 2000
N_STUDY_REPLICATES: int = 200
HALFWIDTH_TOL_PP: float = 1.0
DISPERSION_SENSITIVITY: float = 1.5

# Coverage-dominant scenario: C=4, S=0.5, P=0.5 of the 5.03-point gap.
SCENARIO_C: float = 4.0
SCENARIO_S: float = 0.5
SCENARIO_P: float = 0.5

# Manipulation check (report Sec. "check").
SPREAD_R_FACTOR: float = 0.9
SPREAD_OMEGA_MIN: float = 0.072
SEPARATION_MAX_ABS_CORR: float = 0.3
OVERLAP_R_FACTOR: float = 0.25  # Delta'/4
OVERLAP_OMEGA_TOL: float = 0.02
OVERLAP_MIN_SHARE: float = 0.5
TRANSFER_MIN_RATIO: float = 0.7


def exp5_config(config: dict[str, Any]) -> dict[str, Any]:
    """View this config's stored exp-5 outputs as a standalone exp-5 config."""
    exp5_out = config.get("slurm", {}).get("exp5_outputs")
    if not exp5_out:
        raise ValueError("Config missing slurm.exp5_outputs")
    return {**config, "paths": {"outputs": str(exp5_out)}}


def exp12_config(config: dict[str, Any]) -> dict[str, Any]:
    """View this config's stored exp-12 outputs as a standalone exp-12 config."""
    exp12_out = config.get("slurm", {}).get("exp12_outputs")
    if not exp12_out:
        raise ValueError("Config missing slurm.exp12_outputs")
    return {**config, "paths": {"outputs": str(exp12_out)}}
