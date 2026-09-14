"""Constants and config addressing for the coverage-similarity experiment."""

from __future__ import annotations

from typing import Any

from breadth import N_REPLICATES

from neighbours import N_SPLITS, THRESHOLD_PP

from redundancy import exp5_config

__all__ = [
    "N_SPLITS",
    "N_REPLICATES",
    "THRESHOLD_PP",
    "COHORTS",
    "R_TOL",
    "OMEGA_TOL",
    "NEFF_REL_TOL",
    "MIN_R_GAP",
    "MIN_OMEGA_GAP",
    "TARGET_R_GAP",
    "TARGET_OMEGA_GAP",
    "SEARCH_BASE_SEED",
    "N_RANDOM",
    "N_ANCHORS",
    "N_STARTS",
    "MAX_SWAP_PASSES",
    "DRAW_COUNTS",
    "N_SIM",
    "POWER",
    "EFFECT_PP",
    "DISPERSION_SENSITIVITY",
    "exp5_config",
    "exp12_config",
]

# (patients G, patches per patient m); every cohort receives 160 patches per class.
COHORTS: dict[str, tuple[int, int]] = {
    "good_low": (5, 32),
    "poor_low": (5, 32),
    "good_high": (5, 32),
    "poor_high": (5, 32),
    "ten_match": (10, 16),
}

# Manipulation-check tolerances (report Sec. "check").
R_TOL: float = 0.005
OMEGA_TOL: float = 0.01
NEFF_REL_TOL: float = 0.05
MIN_R_GAP: float = 0.03
MIN_OMEGA_GAP: float = 0.05

# Search targets: margins above the check thresholds (report App. A).
TARGET_R_GAP: float = 0.04
TARGET_OMEGA_GAP: float = 0.07

# Search effort (report App. A).
SEARCH_BASE_SEED: int = 20260914
N_RANDOM: int = 2000
N_ANCHORS: int = 20
N_STARTS: int = 5
MAX_SWAP_PASSES: int = 50

# Precision simulation (report App. B).
DRAW_COUNTS: tuple[int, ...] = (20, 40, 60)
N_SIM: int = 2000
POWER: float = 0.8
EFFECT_PP: float = 2.0
DISPERSION_SENSITIVITY: float = 1.5


def exp12_config(config: dict[str, Any]) -> dict[str, Any]:
    """View this config's stored exp-12 outputs as a standalone exp-12 config."""
    exp12_out = config.get("slurm", {}).get("exp12_outputs")
    if not exp12_out:
        raise ValueError("Config missing slurm.exp12_outputs")
    return {**config, "paths": {"outputs": str(exp12_out)}}
