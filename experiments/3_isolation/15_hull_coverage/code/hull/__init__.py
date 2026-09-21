"""Constants for the hull-coverage experiment (exp-15)."""

from __future__ import annotations

from breadth import N_REPLICATES, N_SPLITS

__all__ = [
    "N_SPLITS",
    "N_REPLICATES",
    "SEARCH_BASE_SEED",
    "N_DRAWS_DEFAULT",
    "DRAW_COUNTS",
    "N_STARTS",
    "MAX_SWAP_PASSES",
    "R_TOL",
    "OMEGA_TOL",
    "SPREAD_R_FACTOR",
    "SPREAD_H_SHARED_FACTOR",
    "SPREAD_H_FULL_FACTOR",
    "SEPARATION_MAX_ABS_CORR",
    "OVERLAP_R_FACTOR",
    "OVERLAP_H_FACTOR",
    "OVERLAP_OMEGA_TOL",
    "OVERLAP_MIN_SHARE",
    "TRANSFER_MIN_RATIO",
]

# Fresh seed: the no-fit feasibility diagnostics of 2026-09-15 used 20260916-20260918.
SEARCH_BASE_SEED: int = 20260920

# Draws per split are a multiple of 35 so the 7-cell (G=5) and 5-cell (G=10) rotations both balance.
N_DRAWS_DEFAULT: int = 35
DRAW_COUNTS: tuple[int, ...] = (35, 70)

# Cohort search effort and loss scales.
N_STARTS: int = 2
MAX_SWAP_PASSES: int = 15
R_TOL: float = 0.005
OMEGA_TOL: float = 0.01

# Manipulation check, fixed before the census runs.
SPREAD_R_FACTOR: float = 0.8
# 0.7 of the half-gap step shared by G=5 and G=10. Lowered from 0.4 on 2026-09-15 after the
# first census reached 93-99 % of 0.4 at G=5 in every split, with every other condition passing
# and before any classifier was trained; reported as a deviation.
SPREAD_H_SHARED_FACTOR: float = 0.35
SPREAD_H_FULL_FACTOR: float = 0.8  # G=5 only: full random-10 to random-5 hull gap
SEPARATION_MAX_ABS_CORR: float = 0.3
OVERLAP_R_FACTOR: float = 0.25
OVERLAP_H_FACTOR: float = 0.125
OVERLAP_OMEGA_TOL: float = 0.02
OVERLAP_MIN_SHARE: float = 0.5
TRANSFER_MIN_RATIO: float = 0.7
