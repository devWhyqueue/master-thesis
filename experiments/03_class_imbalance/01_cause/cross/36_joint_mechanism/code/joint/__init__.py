"""Constants for experiment 36: separating prior-sensitivity from thin-support estimation error.

Four settings cross two interventions (PLAN.md "Controlled experiment"): ``native`` (neither),
``separation_only`` (experiment 34's frozen alpha, no correction), ``centre_only`` (alpha = 1, a
dense same-cohort centre correction), and ``joint`` (both). Each setting fits B/P/S/R
(``separation.FIT_SOURCE``, exp-27/28's balanced/prior-only/support-only/combined-imbalance
reweighting, rho = 100) at G = 10, matching exp-34's cross-dataset support (TCGA-UT reuses
exp-34's own nested ten-patient prefix of its native G = 20 draw; BRACS's native G already is 10).

One shard is one (split, draw): every setting/arm/control shares that draw's own patient cohort,
so a single shard fits all 25 arm evaluations (16 tuned core + 2 wrong-direction + 7 fixed-lambda,
PLAN.md "Execution and genuine precheck") without any cross-shard ordering dependency, since the
fixed-lambda pass only needs that same shard's own native-B tuned result.
"""

from __future__ import annotations

from breadth import N_SPLITS
from prevalence import DEPTH
from separation import ARMS, FIT_SOURCE

__all__ = [
    "N_SPLITS",
    "G",
    "SEVERITY",
    "CENTRE_DEPTH",
    "ARMS",
    "FIT_SOURCE",
    "SETTINGS",
    "SEPARATION_SETTINGS",
    "CORRECTION_SETTINGS",
    "WRONG_DIRECTION_ARMS",
    "FIXED_LAMBDA_ARMS_BY_SETTING",
    "PILOT_DRAWS",
    "MAIN_DRAWS",
    "GATE_INTEGRITY_TOL_PP",
    "GATE_PRIOR_PP",
    "GATE_SUPPORT_PP",
    "GATE_RESCUE_DAMAGE_PP",
    "GATE_RESCUE_ACC_PP",
    "GATE_RESCUE_BALANCED_DROP_MAX_PP",
    "GATE_BOUNDARY_STABILITY_PP",
    "GATE_PRECISION_HALFWIDTH_PP",
    "PRECISION_N_SIM",
    "PRECISION_N_BOOTSTRAP",
    "PRECISION_SEED",
    "SIMULTANEOUS_ALPHA",
]

G: int = (
    10  # patients/class, matched across both datasets (exp-34's own matched support)
)
SEVERITY: int = 100  # the single tested rho, exp-25..35's "ratio 100" severity
CENTRE_DEPTH: int = (
    DEPTH  # 160, PLAN.md's "all 160 eligible patches per selected patient"
)

SETTINGS: tuple[str, ...] = ("native", "separation_only", "centre_only", "joint")
SEPARATION_SETTINGS: tuple[str, ...] = ("separation_only", "joint")  # alpha != 1
CORRECTION_SETTINGS: tuple[str, ...] = (
    "centre_only",
    "joint",
)  # dense-target translation
WRONG_DIRECTION_ARMS: tuple[str, ...] = ("S", "R")  # joint only (PLAN.md line 40)

# Fixed-regularization control (PLAN.md line 39): native's own B needs no separate evaluation,
# since native-B's tuned lambda IS the fixed lambda by definition.
FIXED_LAMBDA_ARMS_BY_SETTING: dict[str, tuple[str, ...]] = {
    "native": ("P", "S", "R"),
    "joint": ("B", "P", "S", "R"),
}

PILOT_DRAWS: tuple[int, ...] = (0, 1)
MAIN_DRAWS: tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8, 9)

# Pilot gate thresholds (PLAN.md "Execution and genuine precheck").
GATE_INTEGRITY_TOL_PP: float = 0.01
GATE_PRIOR_PP: float = 1.0
GATE_SUPPORT_PP: float = 1.0
GATE_RESCUE_DAMAGE_PP: float = 2.0
GATE_RESCUE_ACC_PP: float = 2.0
GATE_RESCUE_BALANCED_DROP_MAX_PP: float = 1.0
GATE_BOUNDARY_STABILITY_PP: float = 0.5
GATE_PRECISION_HALFWIDTH_PP: float = 1.0

PRECISION_N_SIM: int = 2000
PRECISION_N_BOOTSTRAP: int = 500
PRECISION_SEED: int = 20260924

# Bonferroni-adjusted percentile CI for the two simultaneous primary estimates (PLAN.md
# "Report simultaneous 95% intervals for the two primary estimates").
SIMULTANEOUS_ALPHA: float = 0.05
