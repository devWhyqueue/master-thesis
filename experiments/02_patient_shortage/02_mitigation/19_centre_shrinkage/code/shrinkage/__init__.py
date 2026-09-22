"""Constants for the centre-shrinkage experiment (exp-19)."""

from __future__ import annotations

from typing import Any

from centre import PATIENT_COUNTS

__all__ = ["ARMS", "REUSED_ARMS", "ALPHA_FACTORS", "baseline_config"]

ARMS: tuple[str, ...] = tuple(
    f"{family}{g}" for family in ("S", "St") for g in PATIENT_COUNTS
)
REUSED_ARMS: tuple[str, ...] = tuple(
    f"{family}{g}" for family in ("R", "C") for g in PATIENT_COUNTS
)
ALPHA_FACTORS: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 4.0)


def baseline_config(config: dict[str, Any]) -> dict[str, Any]:
    """View this config's stored baseline (exp-16 or exp-18) outputs as a standalone config."""
    baseline_out = config.get("slurm", {}).get("baseline_outputs")
    if not baseline_out:
        raise ValueError("Config missing slurm.baseline_outputs")
    return {**config, "paths": {"outputs": str(baseline_out)}}
