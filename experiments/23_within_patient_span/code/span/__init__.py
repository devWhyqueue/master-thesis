"""Constants for the within-patient span experiment (exp-23)."""

from __future__ import annotations

from centre import PATIENT_COUNTS
from spectrum import baseline_config

__all__ = [
    "ARMS",
    "CAPTURE_MULTIPLES",
    "FIT_FAMILIES",
    "REUSED_ARM_FAMILIES",
    "S_GRID",
    "S_VALUES",
    "baseline_config",
]

# Letter-only family names: ``centre.fit.split_arm`` strips trailing digits. W adds the cohort's own
# within-patient complement directions, R the same eigenvalues on random complement directions.
S_GRID: dict[str, float] = {"a": 0.1, "b": 0.3, "c": 1.0, "d": 3.0}
S_VALUES: tuple[float, ...] = (0.0, *S_GRID.values())
FIT_FAMILIES: tuple[str, ...] = tuple(
    f"{kind}{letter}" for kind in ("W", "R") for letter in S_GRID
)
# Wt/Rt are assembled last from the stored validation scores of the arms before them.
ARMS: tuple[str, ...] = tuple(
    f"{family}{g}" for family in (*FIT_FAMILIES, "Wt", "Rt") for g in PATIENT_COUNTS
)
# Diagnostics measure the top ``m * r_B`` eigenvectors of the arm's covariance.
CAPTURE_MULTIPLES: tuple[int, ...] = (1, 2, 4, 8)

# Reused arm families keyed by the config path pointing at their source experiment's outputs.
REUSED_ARM_FAMILIES: dict[str, tuple[str, ...]] = {
    "baseline_outputs": ("R", "C"),
    "whitening_outputs": ("RW", "RWc"),
}
