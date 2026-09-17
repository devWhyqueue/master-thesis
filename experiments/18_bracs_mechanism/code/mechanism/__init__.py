"""Constants for the BRACS mechanism experiment (exp-18): centre error and patient directions on BRACS."""

from __future__ import annotations

from centre import PATIENT_COUNTS

__all__ = ["ARMS", "CENTRE_FAMILIES"]

CENTRE_FAMILIES: tuple[str, ...] = ("R", "C", "N", "CW")
ARMS: tuple[str, ...] = tuple(
    f"{family}{g}" for family in (*CENTRE_FAMILIES, "RW", "RWc") for g in PATIENT_COUNTS
)
