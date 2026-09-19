"""Constants for the spectrum-regularization experiment (exp-22)."""

from __future__ import annotations

from typing import Any

from centre import PATIENT_COUNTS

__all__ = [
    "ARMS",
    "BETAS",
    "BETA_FAMILIES",
    "REUSED_ARM_FAMILIES",
    "baseline_config",
]

# Letter-only family names: ``centre.fit.split_arm`` strips trailing digits.
BETAS: dict[str, float] = {"F": 0.0, "Ba": 0.25, "Bb": 0.5, "Bc": 0.75}
BETA_FAMILIES: tuple[str, ...] = tuple(BETAS)
# Bt is assembled last from the stored validation scores of the arms before it.
ARMS: tuple[str, ...] = tuple(
    f"{family}{g}"
    for family in (*BETA_FAMILIES, "O", "P", "Bt")
    for g in PATIENT_COUNTS
)

# Reused arm families keyed by the config path pointing at their source experiment's outputs:
# R/C from exp-16 (TCGA-UT) or exp-18 (BRACS, which also holds the pool); RW/RWc from exp-17
# (TCGA-UT) or exp-18 (BRACS).
REUSED_ARM_FAMILIES: dict[str, tuple[str, ...]] = {
    "baseline_outputs": ("R", "C"),
    "whitening_outputs": ("RW", "RWc"),
}


def baseline_config(config: dict[str, Any], key: str) -> dict[str, Any]:
    """View this config's stored source-experiment outputs (``key``) as a standalone config."""
    out = config.get("slurm", {}).get(key)
    if not out:
        raise ValueError(f"Config missing slurm.{key}")
    return {**config, "paths": {"outputs": str(out)}}
