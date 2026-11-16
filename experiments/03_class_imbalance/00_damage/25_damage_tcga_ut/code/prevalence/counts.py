"""Per-class and per-patient patch counts of one prevalence arm (exp-25 allocator)."""

from __future__ import annotations

import numpy as np
from imbalance_benchmark.construction import _adjust_alloc, allocate_counts

from prevalence import G, balanced_of

__all__ = ["class_counts"]


def class_counts(
    arm: str,
    perm: np.ndarray,
    available: list[int],
    pool_counts: list[int],
    g: int = G,
) -> list[int]:
    """Per-class total patch counts for one arm at the shared budget T = g x n x K (n from ``g``)."""
    num_classes = len(available)
    total = g * balanced_of(g) * num_classes
    if arm == "N":
        share = sum(pool_counts)
        target = [total * c / share for c in pool_counts]
        allocated = [min(max(round(t), g), a) for t, a in zip(target, available)]
        _adjust_alloc(allocated, available, target, total - sum(allocated), g)
        return allocated
    ranked = allocate_counts(available, total, float(arm[1:]), g)
    counts = [0] * num_classes
    for rank, cls_idx in enumerate(perm):
        counts[cls_idx] = ranked[rank]
    return counts


def _patient_counts(total: int, g: int = G) -> list[int]:
    """Split one class's total across its g patients; nesting is per-patient (patient_rows), not here."""
    base, remainder = divmod(total, g)
    return [base + 1] * remainder + [base] * (g - remainder)
