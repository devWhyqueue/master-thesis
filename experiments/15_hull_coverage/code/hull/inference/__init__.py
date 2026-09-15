"""Class-recall model, precision simulation, and analysis constants for the hull-coverage experiment."""

from __future__ import annotations

from decomposition import (
    DISPERSION_SENSITIVITY,
    HALFWIDTH_TOL_PP,
    N_SIM,
    N_STUDY_REPLICATES,
)

__all__ = [
    "PATIENT_COUNTS",
    "PREDICTORS",
    "N_SIM",
    "N_STUDY_REPLICATES",
    "HALFWIDTH_TOL_PP",
    "DISPERSION_SENSITIVITY",
    "HULL_DOMINANT_PARTS",
]

# The model's two patient counts: index 0 is G=5 (z=0), index 1 is G=10 (z=1).
PATIENT_COUNTS: tuple[int, int] = (5, 10)
PREDICTORS: tuple[str, ...] = ("r", "h", "omega", "z")

# Hull-dominant scenario of the precision simulation, in percentage points of the 5-to-10 gap.
HULL_DOMINANT_PARTS: dict[str, float] = {"C": 1.5, "H": 3.0, "S": 0.0, "P": 0.5}
