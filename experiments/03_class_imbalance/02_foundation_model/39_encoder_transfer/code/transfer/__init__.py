"""Constants for the encoder-transfer schedule (exp-39, phase 01).

Reuses exp-25/27's arm machinery unchanged: ``B`` is the balanced arm ``r1``, ``R{rho}`` is the
ratio arm ``r{rho}``, and ``P{rho}``/``S{rho}`` reweight ``r1``'s / ``r{rho}``'s own rows toward the
other's class shares (``prevalence.fit._fit_arm``'s ``data_arm``/``prior_arm`` split, exp-27's
pattern). Draws 10-19 extend ``prevalence.PREVALENCE_SEED``'s existing draw sequence past exp-25's
own draws 0-9 (audited for collisions on 2026-09-24 across every exp-25..38 output on Hydra: none
found; see ``configs/protocol_lock.json``). Draw 10000 is reserved for engineering smoke tests.
"""

from __future__ import annotations

__all__ = [
    "RATIOS_NEW",
    "ARMS",
    "FIT_SOURCE",
    "MAIN_DRAWS",
    "SMOKE_DRAW",
    "ENCODERS",
    "LAMBDAS",
    "TOLERANCE",
    "MAX_ITER",
    "TIE_TOLERANCE",
    "BOOTSTRAP_SEED",
    "N_REPLICATES",
]

RATIOS_NEW: tuple[int, ...] = (10, 100)

# Both frozen encoders every fit/analyze shard pairs, per protocol phase 01/04.
ENCODERS: tuple[str, ...] = ("virchow2", "uni2h")

# Frozen lambda grid (protocol_lock.json's readout.lambda_grid): 1e-8..1e2, decade steps, 11
# candidates. Distinct from breadth/decodability's own grids, so the fit stage cannot reuse
# breadth.fit.tune_and_fit_draw's module-level LAMBDAS unchanged (plans/04_implementation.md).
LAMBDAS: tuple[float, ...] = tuple(10.0**p for p in range(-8, 3))
# Solver tolerances: existing, unchanged (breadth's own values, protocol_lock.json).
TOLERANCE: float = 1e-8
MAX_ITER: int = 10000
TIE_TOLERANCE: float = 1e-10

# protocol_lock.json's inference block.
BOOTSTRAP_SEED: int = 39000
N_REPLICATES: int = 10000

ARMS: tuple[str, ...] = (
    ("B",)
    + tuple(f"R{r}" for r in RATIOS_NEW)
    + tuple(f"P{r}" for r in RATIOS_NEW)
    + tuple(f"S{r}" for r in RATIOS_NEW)
)

# (data_arm, prior_arm) in prevalence's r{rho} naming: rows come from data_arm, the loss is
# reweighted toward prior_arm's class shares (None = unweighted).
FIT_SOURCE: dict[str, tuple[str, str | None]] = {
    "B": ("r1", None),
    **{f"R{r}": (f"r{r}", None) for r in RATIOS_NEW},
    **{f"P{r}": ("r1", f"r{r}") for r in RATIOS_NEW},
    **{f"S{r}": (f"r{r}", "r1") for r in RATIOS_NEW},
}

MAIN_DRAWS: tuple[int, ...] = tuple(range(10, 20))
SMOKE_DRAW: int = 10000
