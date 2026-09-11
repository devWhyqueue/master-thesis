"""Constants and config addressing for the multidirectional-redundancy experiment."""

from __future__ import annotations

from typing import Any

__all__ = [
    "TRAINING_BOOTSTRAP_SEED",
    "MEASURES",
    "exp5_config",
]

# Independent of exp5's/exp4's BOOTSTRAP_SEED: seeds the training-patient
# cluster bootstrap, a resampling axis exp5 never performed.
TRAINING_BOOTSTRAP_SEED: int = 20260911

MEASURES: tuple[str, ...] = ("single", "full")


def exp5_config(config: dict[str, Any]) -> dict[str, Any]:
    """View this config's stored exp-5 outputs as a standalone exp-5 config."""
    exp5_out = config.get("slurm", {}).get("exp5_outputs")
    if not exp5_out:
        raise ValueError("Config missing slurm.exp5_outputs")
    return {**config, "paths": {"outputs": str(exp5_out)}}
