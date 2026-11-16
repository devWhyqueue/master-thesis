"""Constants for the prevalence imbalance experiment (exp-25).

Every ratio arm ``r{rho}`` and the native arm ``N`` share the same P patients per class and the
same total patch budget T = P x n x num_classes (n = patches per patient at rho 1); only the
per-class share of T differs. Ratio arms redistribute T across a permuted class rank with exp-02's
exponential-profile allocator (``imbalance_benchmark.construction.allocate_counts``); N keeps the
full train-split's native class shares. The eligibility depth bounds how many patches one patient
can contribute (round-robin over slides), so every class's available budget is P x depth
regardless of arm.

P and n come from ``prevalence.patients_per_class`` / ``prevalence.patches_per_patient``.
``patients_per_class`` returns a ``Quota`` (an ``int`` that also carries n), so the many callers that
thread P as a plain ``g`` need no change. ``G`` / ``BALANCED`` / ``DEPTH`` are the legacy defaults
(P = 20, n = 32, depth 160) that frozen experiments and configs without ``patches_per_patient`` use.
"""

from __future__ import annotations

import math
from typing import Any

from imbalance_benchmark.construction import allocate_counts

__all__ = [
    "ARMS",
    "BALANCED",
    "DEPTH",
    "G",
    "PREVALENCE_SEED",
    "RATIOS",
    "Quota",
    "balanced_of",
    "depth_of",
    "eligibility_depth",
    "patches_per_patient",
    "patients_per_class",
]

G = 20
BALANCED = 32
DEPTH = 160
RATIOS: tuple[int, ...] = (1, 2, 5, 10, 20, 50, 100)
ARMS: tuple[str, ...] = tuple(f"r{r}" for r in RATIOS) + ("N",)
# Fresh seed for this experiment's own patient draws and class-rank permutations.
PREVALENCE_SEED: int = 20260921


class Quota(int):
    """P patients per class that also carries the design's per-patient patch count n."""

    balanced: int
    derive_depth: bool

    def __new__(cls, patients: int, balanced: int, derive_depth: bool) -> "Quota":
        obj = super().__new__(cls, patients)
        obj.balanced = balanced
        obj.derive_depth = derive_depth
        return obj

    def __getnewargs__(self) -> tuple[int, int, bool]:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Keep ``copy`` / ``pickle`` round-trips (configs are deep-copied and sent to workers)."""
        return int(self), self.balanced, self.derive_depth


def balanced_of(g: int) -> int:
    """n of a ``Quota``; legacy ``BALANCED`` for a plain int."""
    return getattr(g, "balanced", BALANCED)


def depth_of(g: int, num_classes: int) -> int:
    """Eligibility depth for P = g: the head class's per-patient need at rho 100.

    ``ceil(head share x T / P)`` with exp-02's allocator head share, so P x depth always covers
    the largest class. Plain ints and configs without ``patches_per_patient`` keep legacy ``DEPTH``.
    """
    if not getattr(g, "derive_depth", False):
        return DEPTH
    total = int(g) * balanced_of(g) * num_classes
    head = allocate_counts([total] * num_classes, total, float(max(RATIOS)), int(g))[0]
    return math.ceil(head / int(g))


def patches_per_patient(config: dict[str, Any]) -> int:
    """n, the per-patient patch count of the balanced arm: config override, else legacy 32."""
    return int(config.get("prevalence", {}).get("patches_per_patient", BALANCED))


def patients_per_class(config: dict[str, Any]) -> Quota:
    """P for this dataset: config override, else legacy 20; carries n from the same config block."""
    block = config.get("prevalence", {})
    return Quota(
        int(block.get("patients_per_class", G)),
        patches_per_patient(config),
        "patches_per_patient" in block,
    )


def eligibility_depth(config: dict[str, Any], num_classes: int) -> int:
    """``depth_of`` for a config's own P and n."""
    return depth_of(patients_per_class(config), num_classes)
