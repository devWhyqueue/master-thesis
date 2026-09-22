"""Constants for the prior-vs-support cause experiment (exp-27).

exp-25's ratio arm ``r{rho}`` moves two things together: the loss's class prior and the tail's
patch support. This experiment separates them per ``rho`` in ``RATIOS_NEW`` (exp-25's ``RATIOS``
without ``rho = 1``, which is the shared balanced baseline):

- ``P{rho}`` (prior only): r1's balanced rows, reweighted toward r{rho}'s class shares.
- ``S{rho}`` (support only): r{rho}'s rows, reweighted back to r1's balanced shares.
- ``LP{rho}``/``Lr{rho}``: post-hoc logit adjustment of ``P{rho}``/``r{rho}``'s stored test
  probabilities (subtract each class's log prior share); no solver fit.

``r1`` and ``r{rho}`` themselves are reused from exp-25's stored outputs (fully paired: same
splits, draws, patients, class permutation, and nested patch rows), addressed through
``spectrum.baseline_config`` against ``slurm.prevalence_outputs``.
"""

from __future__ import annotations

from prevalence import RATIOS

__all__ = [
    "RATIOS_NEW",
    "P_ARMS",
    "S_ARMS",
    "LP_ARMS",
    "LR_ARMS",
    "NEW_FIT_ARMS",
    "ADJUSTED_ARMS",
    "REUSED_ARMS",
    "ARMS",
    "FIT_SOURCE",
    "ADJUST_SOURCE",
]

RATIOS_NEW: tuple[int, ...] = RATIOS[1:]  # (2, 5, 10, 20, 50, 100)

P_ARMS: tuple[str, ...] = tuple(f"P{r}" for r in RATIOS_NEW)
S_ARMS: tuple[str, ...] = tuple(f"S{r}" for r in RATIOS_NEW)
LP_ARMS: tuple[str, ...] = tuple(f"LP{r}" for r in RATIOS_NEW)
LR_ARMS: tuple[str, ...] = tuple(f"Lr{r}" for r in RATIOS_NEW)

# Fit stage: 2 arms x 6 ratios x (3 splits x 10 draws) = 360 solver fits.
NEW_FIT_ARMS: tuple[str, ...] = P_ARMS + S_ARMS
# Derived at fit time from already-stored test probabilities; no solver fit.
ADJUSTED_ARMS: tuple[str, ...] = LP_ARMS + LR_ARMS
# Reused from exp-25's own outputs (slurm.prevalence_outputs), unweighted, unchanged.
REUSED_ARMS: tuple[str, ...] = ("r1",) + tuple(f"r{r}" for r in RATIOS_NEW)

ARMS: tuple[str, ...] = REUSED_ARMS + NEW_FIT_ARMS + ADJUSTED_ARMS

# (data_arm, prior_arm) for each new fit arm: rows come from data_arm, the loss is reweighted
# toward prior_arm's class shares (prevalence.fit._fit_arm / _prior_weights).
FIT_SOURCE: dict[str, tuple[str, str]] = {
    **{f"P{r}": ("r1", f"r{r}") for r in RATIOS_NEW},
    **{f"S{r}": (f"r{r}", "r1") for r in RATIOS_NEW},
}

# (source_arm, own_or_exp25) for each adjusted arm: which fit's stored test probabilities to
# adjust, and whether that fit lives in this experiment's own outputs ("own", Pρ, whose
# prior_counts prevalence.fit._fit_arm now stores) or exp-25's (rρ, whose own realized
# class_counts stand in for its implicit training prior).
ADJUST_SOURCE: dict[str, tuple[str, str]] = {
    **{f"LP{r}": (f"P{r}", "own") for r in RATIOS_NEW},
    **{f"Lr{r}": (f"r{r}", "exp25") for r in RATIOS_NEW},
}
