"""Constants for the patient-directions experiment (exp-17)."""

from __future__ import annotations

from typing import Any

from centre import PATIENT_COUNTS

__all__ = ["ARMS", "KAPPA_FACTORS", "REUSED_ARMS", "exp16_config"]

ARMS: tuple[str, ...] = tuple(
    f"{family}{g}" for family in ("RW", "RWc") for g in PATIENT_COUNTS
)
# exp-16's grid plus 0.1: the smallest exp-16 factor won in 174/180 RW/RWc fits.
KAPPA_FACTORS: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0)
REUSED_ARMS: tuple[str, ...] = tuple(
    f"{family}{g}" for family in ("R", "C", "CW") for g in PATIENT_COUNTS
)


def exp16_config(config: dict[str, Any]) -> dict[str, Any]:
    """View this config's stored exp-16 outputs as a standalone exp-16 config."""
    exp16_out = config.get("slurm", {}).get("exp16_outputs")
    if not exp16_out:
        raise ValueError("Config missing slurm.exp16_outputs")
    return {**config, "paths": {"outputs": str(exp16_out)}}
