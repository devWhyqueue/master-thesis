"""Constants for the oracle weighting experiment (exp-24)."""

from __future__ import annotations

from spectrum import baseline_config

__all__ = ["ARMS", "G", "REUSED_ARM_FAMILIES", "baseline_config"]

# Letter-only family names: ``centre.fit.split_arm`` strips trailing digits. All arms carry the cohort block
# (U, lambda) plus a complement block weighted by the pool's between-patient variance along each direction:
# Wo the cohort's within-patient directions V, Ro random complement directions, Po the pool's own
# top eigenvectors of the complement covariance.
G = 5
ARMS: tuple[str, ...] = tuple(f"{family}{G}" for family in ("Wo", "Ro", "Po"))

# Reused arm families keyed by the config path pointing at their source experiment's outputs.
REUSED_ARM_FAMILIES: dict[str, tuple[str, ...]] = {
    "baseline_outputs": ("R", "C"),
    "whitening_outputs": ("RW", "RWc"),
    "span_outputs": ("Wt", "Rt"),
}
