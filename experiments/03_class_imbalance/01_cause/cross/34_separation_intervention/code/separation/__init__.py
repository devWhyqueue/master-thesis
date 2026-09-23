"""Constants for the class-separation intervention experiment (exp-34).

x^(alpha) = x + (alpha - 1)(mu_y - mu_bar): shift every class's rows by a fixed multiple of its
training-only, patient-balanced centre's offset from the grand mean of centres (``geometry.py``).
Alpha expands BRACS toward TCGA-UT's separation, or contracts TCGA-UT toward BRACS's; frozen at
``precheck`` from each dataset's own read-only separation index J = median nearest-centre
distance / pooled within-class RMS, computed at exp-25/26's DEPTH = 160 pool (``prevalence.DEPTH``,
not exp-16's own POOL_DEPTH = 32).

Four arms per setting reuse exp-27/28's B/P/S/R-equivalent reweighting (``prevalence.fit``):
B = balanced rows, balanced prior (unweighted r1). P = balanced rows, reweighted toward the
imbalanced prior. S = imbalanced rows, reweighted back to balanced. R = imbalanced rows,
imbalanced prior (unweighted r100). "Imbalanced" is rho = 100, exp-25/26/27/28's own severity.

Settings: BRACS fits ``expanded_bracs`` (new) plus a pilot-only ``native_bracs_replay`` (gate 1,
reproduction). TCGA-UT fits ``tcga_native_10`` and ``tcga_contracted_10`` (both new: exp-25/27's
native run is G = 20, so the G = 10 match-support control needs its own fit). "TCGA-UT's 10
patients are the first 10 of each existing paired 20-patient draw" (PLAN.md): ``fit.py`` re-derives
that draw deterministically from ``prevalence.fit._draw_patients`` rather than storing it twice.
"""

from __future__ import annotations

from breadth import N_SPLITS
from prevalence import DEPTH

__all__ = [
    "N_SPLITS",
    "CENTRE_DEPTH",
    "G",
    "SEVERITY",
    "ARMS",
    "FIT_SOURCE",
    "PILOT_DRAWS",
    "MAIN_DRAWS",
    "NEW_SETTINGS_BY_DATASET",
    "PILOT_ONLY_SETTINGS_BY_DATASET",
    "pilot_settings",
    "main_settings",
    "ALPHA_BOUNDS",
    "GATE_REPRODUCTION_TOL_PP",
    "GATE_MANIPULATION_PP",
    "GATE_TRIVIAL_BOUNDS",
    "GATE_MECHANISM_PP",
    "GATE_POWER_TARGET",
    "PRECISION_EFFECT_PP",
    "PRECISION_N_SIM",
    "PRECISION_N_BOOTSTRAP",
    "PRECISION_SEED",
]

CENTRE_DEPTH: int = DEPTH  # 160, exp-25/26's pool depth
G: int = 10  # patients/class, matched across both datasets (exp-28's native BRACS G)
SEVERITY: int = 100  # the single tested rho, exp-25..33's "ratio 100" severity

ARMS: tuple[str, ...] = ("B", "P", "S", "R")
# (data_arm, prior_arm) fed to prevalence.fit.class_counts/_prior_weights; None = unweighted.
FIT_SOURCE: dict[str, tuple[str, str | None]] = {
    "B": ("r1", None),
    "P": ("r1", f"r{SEVERITY}"),
    "S": (f"r{SEVERITY}", "r1"),
    "R": (f"r{SEVERITY}", None),
}

PILOT_DRAWS: tuple[int, ...] = (0,)
MAIN_DRAWS: tuple[int, ...] = (1, 2, 3, 4)

# Settings requiring new fits, by dataset, and the extra pilot-only reproduction replay.
NEW_SETTINGS_BY_DATASET: dict[str, tuple[str, ...]] = {
    "bracs": ("expanded_bracs",),
    "tcga_ut": ("tcga_native_10", "tcga_contracted_10"),
}
PILOT_ONLY_SETTINGS_BY_DATASET: dict[str, tuple[str, ...]] = {
    "bracs": ("native_bracs_replay",),
    "tcga_ut": (),
}


def pilot_settings(dataset: str) -> tuple[str, ...]:
    """This dataset's pilot-phase settings: its new settings plus any replay control."""
    return NEW_SETTINGS_BY_DATASET[dataset] + PILOT_ONLY_SETTINGS_BY_DATASET[dataset]


def main_settings(dataset: str) -> tuple[str, ...]:
    """This dataset's main-phase settings: new settings only (controls are reused, not refit)."""
    return NEW_SETTINGS_BY_DATASET[dataset]


ALPHA_BOUNDS: tuple[float, float] = (1.25, 4.0)

GATE_REPRODUCTION_TOL_PP: float = 0.01
GATE_MANIPULATION_PP: float = 3.0
GATE_TRIVIAL_BOUNDS: tuple[float, float] = (25.0, 90.0)
GATE_MECHANISM_PP: float = 2.0
GATE_POWER_TARGET: float = 0.80

# Precision simulation (gate 4): project power for the pre-registered 2 pp mechanism effect at
# the main experiment's design (3 splits x 4 draws). Both primary comparisons use a plain 95%
# bootstrap CI (lower bound > 0) as the significance criterion, which is the Holm-adjusted
# one-sided 0.025 worst case for two tests -- conservative for whichever test ends up larger.
PRECISION_EFFECT_PP: float = GATE_MECHANISM_PP
PRECISION_N_SIM: int = 2000
PRECISION_N_BOOTSTRAP: int = 500
PRECISION_SEED: int = 20260924
