"""Constants for experiment 37: does coverage-maximizing selection recover damage from thin
support, where exp-33..36 found the prior channel but left 3.29 vs 0.53 pp of matched D_S
unexplained?

Regime: exp-36 native setting, unchanged (PLAN.md "Regime"). G = 10, 32 balanced patches/patient
(320/class), rho = 100, frozen Virchow2, 3 splits x draws 2-9. One shard is one (split, draw); every
class in it has a frozen 320-row B pool (32/patient) and an S100-realized total, capped at that
pool's size so ``S ⊂ B`` holds for every class, not only the thinned tail (see
``support.shard.class_pools``). Three S arms reselect S100's own per-class, per-patient counts from
within that same pool: ``random`` (S100's own nested-prefix rows), ``coverage`` (greedy quota
facility location, maximizing coverage of the full pool), and ``redundant`` (per-patient nearest to
the patient-balanced pool centre).
"""

from __future__ import annotations

from breadth import N_SPLITS
from prevalence import BALANCED, DEPTH, G

__all__ = [
    "N_SPLITS",
    "G",
    "BALANCED",
    "DEPTH",
    "SEVERITY",
    "S_ARMS",
    "ARMS",
    "MAIN_DRAWS",
    "GATE0_SUPPORT_GAP_PP",
    "GATE1_NS",
    "GATE1_TAIL_N",
    "GATE1_RATIO",
    "GATE2_COVERAGE_MAX_RATIO",
    "GATE2_REDUNDANT_MIN_RATIO",
    "LABEL_RESCUE_HALF",
]

SEVERITY: int = 100  # the single tested rho, PLAN.md "Regime"

S_ARMS: tuple[str, ...] = ("random", "coverage", "redundant")
ARMS: tuple[str, ...] = ("B", *S_ARMS)

# Locked draws 2-9 (PLAN.md "Regime"): the same 8 untouched draws exp-36's own gate 2 used.
MAIN_DRAWS: tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8, 9)

# Gate 0 -- premise (PLAN.md "Gate 0"): matched native D_S(BRACS) - D_S(TCGA) at fixed lambda.
GATE0_SUPPORT_GAP_PP: float = 1.0

# Gate 1 -- coverage gap (PLAN.md "Gate 1"): random-subset curve c(n), n=12 is the BRACS S100 tail
# count.
GATE1_NS: tuple[int, ...] = (3, 6, 12, 26, 56, 120)
GATE1_TAIL_N: int = 12
GATE1_RATIO: float = 1.2

# Gate 2 -- manipulation check (PLAN.md "Gate 2"), on thinned classes.
GATE2_COVERAGE_MAX_RATIO: float = 0.75
GATE2_REDUNDANT_MIN_RATIO: float = 1.25

# Labels (PLAN.md "Labels"): G_coverage <= 0.5 x G_random for coverage_explains.
LABEL_RESCUE_HALF: float = 0.5
