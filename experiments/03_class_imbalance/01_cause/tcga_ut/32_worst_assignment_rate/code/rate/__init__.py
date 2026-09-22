"""Constants for the rate-weighted worst tail-class assignment experiment (exp-32).

Exp-31's directed pair-separation score row-normalized the off-diagonal confusion, so every class
carried total weight 1 regardless of its error rate: an easy class with almost no errors counted as
much as a heavily confused one. That let the score reward rank/headroom monotonicity rather than
the damage mechanism, which is why exp-31's G2 failed and Part B was never submitted. Exp-32 keeps
the same score formula and search (``worst.score``) but replaces the weight with the absolute
confusion rate ``w_cd = P(pred d | true c)`` (``rate.order.confusion_rates``), which passes the
same pre-registered gate that broke exp-31 (``rate.precheck``).

Part A (0 new fits, run before this plan) validated the rate score against exp-25's 30 random r100
draws and exp-30's easy/hard anchors, and fit a calibration line predicting damage from the score.
Part B fits the worst (argmax S_rate), flip (reversed worst, direction control), and mild (argmin
S_rate) orders at rho 100 (90 new fits, 3 per (split, draw) shard).
"""

from __future__ import annotations

__all__ = [
    "RATIO_SEARCH",
    "NEW_FIT_ARMS",
    "REUSED_ARMS",
    "EXP30_ARMS",
    "ARM_ORDER",
    "SEARCH_SEED",
    "N_RESTARTS",
]

RATIO_SEARCH: int = 100

NEW_FIT_ARMS: tuple[str, ...] = (
    f"worst_r{RATIO_SEARCH}",
    f"flip_r{RATIO_SEARCH}",
    f"mild_r{RATIO_SEARCH}",
)
# Reused unchanged from exp-25's stored outputs (prevalence_outputs).
REUSED_ARMS: tuple[str, ...] = ("r1", "r10", "r100", "N")
# Reused unchanged from exp-30's stored outputs (tail_outputs).
EXP30_ARMS: tuple[str, ...] = ("easy_r100", "hard_r100")

# Which derived rank order (``rate.order.derive_orders`` key) each new arm fits; all at rho 100.
ARM_ORDER: dict[str, str] = {
    f"worst_r{RATIO_SEARCH}": "worst",
    f"flip_r{RATIO_SEARCH}": "flip",
    f"mild_r{RATIO_SEARCH}": "mild",
}

# Same seed and restart count as exp-31's local search.
SEARCH_SEED: int = 20260923
N_RESTARTS: int = 50
