"""Constants for the worst tail-class assignment experiment (exp-31).

Exp-30 asked whether the class-to-rank assignment matters and found that difficulty (pooled r1
recall) is not the deciding factor. Exp-31 fits the assignment exp-30's discussion names as the
natural worst-case candidate: one directed pair-separation score ``S`` (``worst.score``), built
from each class's confusion partners (``assignment.properties._confusion_counts``) and headroom
(pooled r1 recall), maximized/minimized by local search over the 30! class orders to find a worst
(``sep``) and mildest (``tog``) assignment. ``flip`` is ``sep`` reversed: a direction control with
identical undirected rank-gap separation but every within-pair role flipped.

Part A (0 new fits) sanity-checks ``S`` against exp-30's already-observed easy/hard/random damage
before any new fit is spent (``worst.precheck``). Part B fits ``sep``/``flip``/``tog`` at rho 100
and ``sep`` again at rho 10 (120 new fits total, 4 per (split, draw) shard).
"""

from __future__ import annotations

__all__ = [
    "RATIO_SEARCH",
    "RATIO_LOW",
    "NEW_FIT_ARMS",
    "REUSED_ARMS",
    "EXP30_ARMS",
    "ARM_ORDER",
    "ARM_RATIO",
    "CONFUSION_PAIRS",
    "SEARCH_SEED",
    "N_RESTARTS",
]

RATIO_SEARCH: int = 100
RATIO_LOW: int = 10

NEW_FIT_ARMS: tuple[str, ...] = (
    f"sep_r{RATIO_SEARCH}",
    f"flip_r{RATIO_SEARCH}",
    f"tog_r{RATIO_SEARCH}",
    f"sep_r{RATIO_LOW}",
)
# Reused unchanged from exp-25's stored outputs (prevalence_outputs).
REUSED_ARMS: tuple[str, ...] = ("r1", "r10", "r100", "N")
# Reused unchanged from exp-30's stored outputs (tail_outputs).
EXP30_ARMS: tuple[str, ...] = ("easy_r100", "hard_r100")

# Which derived rank order (``worst.order.derive_orders`` key) and ratio each new arm fits.
ARM_ORDER: dict[str, str] = {
    f"sep_r{RATIO_SEARCH}": "sep",
    f"flip_r{RATIO_SEARCH}": "flip",
    f"tog_r{RATIO_SEARCH}": "tog",
    f"sep_r{RATIO_LOW}": "sep",
}
ARM_RATIO: dict[str, int] = {
    f"sep_r{RATIO_SEARCH}": RATIO_SEARCH,
    f"flip_r{RATIO_SEARCH}": RATIO_SEARCH,
    f"tog_r{RATIO_SEARCH}": RATIO_SEARCH,
    f"sep_r{RATIO_LOW}": RATIO_LOW,
}

# The confusion pairs exp-30's discussion names (KIRC/KIRP/KICH expanded to all 3 pairs).
CONFUSION_PAIRS: tuple[tuple[str, str], ...] = (
    ("Colon_adenocarcinoma", "Rectum_adenocarcinoma"),
    ("Head_and_Neck_squamous_cell_carcinoma", "Lung_squamous_cell_carcinoma"),
    ("Glioblastoma_multiforme", "Brain_Lower_Grade_Glioma"),
    ("Kidney_renal_clear_cell_carcinoma", "Kidney_renal_papillary_cell_carcinoma"),
    ("Kidney_renal_clear_cell_carcinoma", "Kidney_Chromophobe"),
    ("Kidney_renal_papillary_cell_carcinoma", "Kidney_Chromophobe"),
)

# Fresh seed for this experiment's own local search.
SEARCH_SEED: int = 20260923
N_RESTARTS: int = 50
