"""Constants for the centre-error experiment (exp-16)."""

from __future__ import annotations

from breadth import N_SPLITS

__all__ = [
    "N_SPLITS",
    "N_DRAWS",
    "COHORT_SEED",
    "PATCH_BUDGET",
    "POOL_DEPTH",
    "PATIENT_COUNTS",
    "KAPPA_FACTORS",
    "ARMS",
    "SHIFT_TOL",
    "patches_per_patient",
]

N_DRAWS: int = 10
# Fresh seed: the split-0 pre-checks of 2026-09-16 used 20260930.
COHORT_SEED: int = 20261001
PATCH_BUDGET: int = 160
POOL_DEPTH: int = 32
PATIENT_COUNTS: tuple[int, ...] = (5, 10, 20)
# Whitening strengths in units of 1 / mean between-patient eigenvalue.
KAPPA_FACTORS: tuple[float, ...] = (1.0, 10.0, 100.0)
ARMS: tuple[str, ...] = (
    *(f"{family}{g}" for family in ("R", "C", "N", "CW") for g in PATIENT_COUNTS),
    "Swap5",
    "Glob5",
    "Disc5",
    "Off5",
)
SHIFT_TOL: float = 1e-6


def patches_per_patient(g: int) -> int:
    """Patches per patient at patient count ``g`` under the fixed per-class budget."""
    return PATCH_BUDGET // g
