"""Constants for the centre-uncertainty experiment (exp-21)."""

from __future__ import annotations

from breadth import LAMBDAS, MAX_ITER, TIE_TOLERANCE, TOLERANCE

from centre import PATIENT_COUNTS

from directional import baseline_arm_dir, baseline_arm_score, baseline_config

__all__ = [
    "ARMS",
    "COVARIANCE_KINDS",
    "FIXED_T",
    "LAMBDAS",
    "MAX_ITER",
    "NONZERO_T_FACTORS",
    "REUSED_ARM_FAMILIES",
    "T_FACTORS",
    "TIE_TOLERANCE",
    "TOLERANCE",
    "baseline_arm_dir",
    "baseline_arm_score",
    "baseline_config",
]

ARMS: tuple[str, ...] = tuple(
    f"{family}{g}" for family in ("U", "Ut", "It") for g in PATIENT_COUNTS
)
# Covariance behind each tuned arm; the fixed-strength arm U reads the patient candidates.
COVARIANCE_KINDS: tuple[str, ...] = ("patient", "isotropic")
T_FACTORS: tuple[float, ...] = (0.0, 0.25, 1.0, 4.0)
# t = 0 is never fit: it is ordinary logistic regression on unchanged features, i.e. the R arm.
NONZERO_T_FACTORS: tuple[float, ...] = tuple(t for t in T_FACTORS if t)
FIXED_T: float = 1.0

# Reused arm families keyed by the config path pointing at their source experiment's outputs.
REUSED_ARM_FAMILIES: dict[str, tuple[str, ...]] = {
    "baseline_outputs": ("R", "C"),
    "exp19_outputs": ("S", "St"),
    "exp20_outputs": ("A", "At"),
    "whitening_outputs": ("RWc",),
}
