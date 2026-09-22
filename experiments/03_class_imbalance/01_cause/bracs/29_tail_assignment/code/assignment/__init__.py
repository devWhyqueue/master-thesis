"""Constants for the tail-class assignment experiment (exp-29).

exp-26's ratio arms r{rho} permute the class-to-rank assignment once per (split, draw) via
``prevalence.fit.class_permutation``. This experiment additionally cycles that same draw's
permutation through its other 6 rotations (``SHIFTS``), so every one of BRACS's 7 classes sits
in the tail exactly once per draw (a Latin square over class x rank), instead of only ~4 times
across exp-26's 30 independent random draws.

Shift ``k = 0`` is exp-26's own permutation, so ``r{rho}`` (and ``r1``, the shared balanced
baseline) are reused unchanged from exp-26's stored outputs (``slurm.prevalence_outputs``), not
refit. Shifts ``k = 1..6`` are new fits, named ``a{k}_r{rho}``.
"""

from __future__ import annotations

from prevalence import RATIOS as _PREVALENCE_RATIOS

__all__ = [
    "RATIOS",
    "SHIFTS",
    "NEW_FIT_ARMS",
    "REUSED_ARMS",
    "ARMS",
    "ATYPICAL",
]

RATIOS: tuple[int, ...] = (10, 100)
assert set(RATIOS) <= set(_PREVALENCE_RATIOS)

SHIFTS: range = range(1, 7)  # k = 0 is exp-26's own permutation, reused as r{rho}

# Fit stage: 6 shifts x 2 ratios x (3 splits x 10 draws) = 360 solver fits.
NEW_FIT_ARMS: tuple[str, ...] = tuple(f"a{k}_r{r}" for r in RATIOS for k in SHIFTS)
# Reused from exp-26's own outputs (slurm.prevalence_outputs), unchanged.
REUSED_ARMS: tuple[str, ...] = ("r1",) + tuple(f"r{r}" for r in RATIOS)

ARMS: tuple[str, ...] = REUSED_ARMS + NEW_FIT_ARMS

# Pre-specified contrast (design review): BRACS's overlapping atypical classes vs the rest.
ATYPICAL: tuple[str, ...] = ("FEA", "ADH")
