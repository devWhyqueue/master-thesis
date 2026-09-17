"""Constants for the patient-directions experiment (exp-17)."""

from __future__ import annotations

from typing import Any

from centre import PATIENT_COUNTS

__all__ = ["ARMS", "REUSED_ARMS", "exp16_config"]

ARMS: tuple[str, ...] = tuple(
    f"{family}{g}" for family in ("RW", "RWc") for g in PATIENT_COUNTS
)
REUSED_ARMS: tuple[str, ...] = tuple(
    f"{family}{g}" for family in ("R", "C", "CW") for g in PATIENT_COUNTS
)


def exp16_config(config: dict[str, Any]) -> dict[str, Any]:
    """View this config's stored exp-16 outputs as a standalone exp-16 config."""
    exp16_out = config.get("slurm", {}).get("exp16_outputs")
    if not exp16_out:
        raise ValueError("Config missing slurm.exp16_outputs")
    return {**config, "paths": {"outputs": str(exp16_out)}}
