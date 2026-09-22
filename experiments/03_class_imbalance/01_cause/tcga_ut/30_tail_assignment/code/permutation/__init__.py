"""Constants for the tail-class assignment relevance experiment (exp-30).

Part A reuses exp-25's stored r1/r10/r100/N fits unchanged (30 independent random class
orders, one per split x draw draw) to fit a per-class piecewise own-recall-vs-allocation model
(``permutation.model``), then samples random permutations from it. Part B confirms the reading
with two new r100 fits per (split, draw): the easy-tail order (``easy_r100``, exp-25's pooled r1
test recall highest class placed in the tail) and the hard-tail order (``hard_r100``, that
placement reversed). ``RATIO_B`` is the only ratio Part B fits; ``MODEL_RATIOS`` are the ratios
Part A's sampled-permutation spread is reported at.
"""

from __future__ import annotations

__all__ = [
    "RATIO_B",
    "NEW_FIT_ARMS",
    "REUSED_ARMS",
    "MODEL_RATIOS",
    "N_SAMPLED",
    "SAMPLE_SEED",
]

RATIO_B: int = 100
NEW_FIT_ARMS: tuple[str, ...] = (f"easy_r{RATIO_B}", f"hard_r{RATIO_B}")
REUSED_ARMS: tuple[str, ...] = ("r1", "r10", "r100", "N")
MODEL_RATIOS: tuple[int, ...] = (10, 100)
N_SAMPLED: int = 10_000
# Fresh seed for this experiment's own sampled-permutation draws.
SAMPLE_SEED: int = 20260922
