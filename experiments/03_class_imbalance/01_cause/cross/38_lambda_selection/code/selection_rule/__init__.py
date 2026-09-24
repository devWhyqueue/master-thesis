"""Constants for experiment 38: how much of exp-36's prior-channel gap comes from lambda
selection, not the prior mechanism itself.

Regime: exp-36 native setting, unchanged. G = 10, rho = 100, frozen Virchow2, 3 splits. Every
(split, draw) shard's native B/P/S/R grids (``joint.grid.write_grid``, every lambda retained) are
re-predicted under four rules -- ``tuned`` (validation-selected, exp-36's own choice), ``oracle``
(cross-fit: lambda picked on one test-patient fold, scored on the other, swapped and concatenated
-- an upper bound, not deployable), ``naive`` (lambda picked and scored on the same test patients --
winner's curse, descriptive only), and ``fixed`` (every arm re-evaluated at native B's own tuned
lambda, exp-36's own fixed-regularization control) -- with no refit at all.
"""

from __future__ import annotations

from breadth import N_SPLITS
from joint import ARMS, CENTRE_DEPTH, G

__all__ = [
    "N_SPLITS",
    "G",
    "CENTRE_DEPTH",
    "ARMS",
    "RULES",
    "PILOT_DRAWS",
    "MAIN_DRAWS",
    "CLOSURE_GATE",
    "ORACLE_FOLD_SEED",
    "INTEGRITY_TOL_PP",
    "LABEL_NO_PRIOR_GAP",
    "LABEL_SELECTION_EXPLAINS",
    "LABEL_SELECTION_CONTRIBUTES",
    "LABEL_PRIOR_GAP_INTRINSIC",
]

RULES: tuple[str, ...] = ("tuned", "oracle", "naive", "fixed")

PILOT_DRAWS: tuple[int, ...] = (
    0,
    1,
)  # matches exp-36's own pilot draws, exact reproduction
MAIN_DRAWS: tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8, 9)

CLOSURE_GATE: float = 0.5  # phase-1 gate: cross-fit closure on draws 0-1

ORACLE_FOLD_SEED: int = (
    20260924  # fixed cross-fit fold seed (class-stratified test-patient split)
)

# Phase-1 integrity check: re-derived D_P must reproduce exp-36's stored values to this tolerance.
INTEGRITY_TOL_PP: float = 0.0001

LABEL_NO_PRIOR_GAP = "no_prior_gap"
LABEL_SELECTION_EXPLAINS = "selection_explains"
LABEL_SELECTION_CONTRIBUTES = "selection_contributes"
LABEL_PRIOR_GAP_INTRINSIC = "prior_gap_intrinsic"
