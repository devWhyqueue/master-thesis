"""Constants for the prevalence imbalance experiment (exp-25).

Every ratio arm ``r{rho}`` and the native arm ``N`` share the same G = 20 patients per class and the
same total patch budget T = G x BALANCED x num_classes; only the per-class share of T differs. Ratio
arms redistribute T across a permuted class rank with exp-02's exponential-profile allocator
(``imbalance_benchmark.construction.allocate_counts``); N keeps the full train-split's native class
shares. DEPTH bounds how many patches one patient can contribute (round-robin over slides), so every
class's available budget is G x DEPTH regardless of arm.
"""

from __future__ import annotations

__all__ = ["ARMS", "BALANCED", "DEPTH", "G", "PREVALENCE_SEED", "RATIOS"]

G = 20
BALANCED = 32
DEPTH = 160
RATIOS: tuple[int, ...] = (1, 2, 5, 10, 20, 50, 100)
ARMS: tuple[str, ...] = tuple(f"r{r}" for r in RATIOS) + ("N",)
# Fresh seed for this experiment's own patient draws and class-rank permutations.
PREVALENCE_SEED: int = 20260921
