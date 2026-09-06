"""Fit work-item enumeration and SLURM-array sharding."""

from __future__ import annotations

from dataclasses import dataclass

from diversity.manifests import ALLOCATIONS

__all__ = [
    "METHODS",
    "FIT_LEVELS",
    "N_SEEDS",
    "FitUnit",
    "fit_units",
    "resolve_fit_bundle",
]

# The three arms the report's Table "arms" specifies for exp-3 (Sec. "Grid,
# arms, and budget"): ce is the damage baseline, weighted_ce the unmatched
# (prevalence) control, semantic_scale_ce the matched (diversity) arm.
METHODS = ("ce", "weighted_ce", "semantic_scale_ce")
# semantic_scale_ce carries an SsbPool re-encode cost (plan "Sharding"), so
# it is scheduled as its own group with its own, lighter packing.
STANDARD_METHODS = ("ce", "weighted_ce")
SEMANTIC_SCALE_METHODS = ("semantic_scale_ce",)
FIT_LEVELS = ("narrow", "wide")  # 'random' is imported, never fitted.
N_SEEDS = 5


@dataclass(frozen=True)
class FitUnit:
    """One (split, level, allocation, method, seed) confirmation fit."""

    split_index: int
    level: str
    allocation: str
    method: str
    seed_index: int


def _group_methods(group: str) -> tuple[str, ...]:
    if group == "standard":
        return STANDARD_METHODS
    if group == "semantic_scale":
        return SEMANTIC_SCALE_METHODS
    raise ValueError(f"Unknown fit group: {group}")


def fit_units(group: str) -> list[FitUnit]:
    """Every work item for one method group, in a stable deterministic order."""
    return [
        FitUnit(split_index, level, allocation, method, seed_index)
        for split_index in range(3)
        for level in FIT_LEVELS
        for allocation in ALLOCATIONS
        for method in _group_methods(group)
        for seed_index in range(N_SEEDS)
    ]


def resolve_fit_bundle(
    group: str, shard_index: int, shards_per_task: int
) -> list[FitUnit]:
    """The slice of one group's work items one SLURM array task owns."""
    units = fit_units(group)
    start = shard_index * shards_per_task
    return units[start : start + shards_per_task]
