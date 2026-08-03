from __future__ import annotations

from dataclasses import dataclass

from imbalance_benchmark.modeling.context import group_conditions, roster_for_condition
from imbalance_benchmark.modeling.workflows.tuning.tuning_schedule import (
    bundled_array_size,
)

__all__ = [
    "CONFIRMATION_SEED_COUNT",
    "ConfirmUnit",
    "confirm_group_methods",
    "confirm_units_for_group",
    "confirm_array_size",
    "resolve_confirm_bundle",
]

CONFIRMATION_SEED_COUNT = 5


@dataclass(frozen=True)
class ConfirmUnit:
    """One independently fittable (split, condition, method, seed) slice.

    Tail assignments are not a dimension here: a unit still fits every
    assignment scoped to its condition internally, exactly as the monolithic
    confirmation loop already does, since the assignment count (2 or 3) is
    only known from the per-split frozen manifest at run time.
    """

    split_index: int
    condition: str
    method: str
    seed_index: int


def confirm_group_methods(is_mil: bool, condition: str) -> tuple[str, ...]:
    """One condition's roster methods scheduled as their own units.

    Post-hoc logit adjustment is excluded: it has no independent unit because
    it inherits its seed's CE checkpoint in-memory and rides with that seed's
    "ce" unit (checkpoints are never persisted to disk between units).
    """
    return tuple(
        method
        for method in roster_for_condition(is_mil, condition)
        if method != "post_hoc_logit_adjustment"
    )


def confirm_units_for_group(
    group: str, is_mil: bool, splits: tuple[int, ...] = (0, 1, 2)
) -> list[ConfirmUnit]:
    """Enumerate every independently schedulable unit for one partition group.

    Ordering is stable and deterministic (split, then condition, then method,
    then seed) and depends only on the roster and regime, never on frozen
    per-dataset state, so a SLURM array index resolves to the same unit both
    when the workflow is built locally and when a task runs on the cluster.
    """
    return [
        ConfirmUnit(split_index, condition, method, seed_index)
        for split_index in splits
        for condition in group_conditions(group)
        for method in confirm_group_methods(is_mil, condition)
        for seed_index in range(CONFIRMATION_SEED_COUNT)
    ]


def confirm_array_size(group: str, is_mil: bool, shards_per_task: int) -> int:
    """Return the SLURM array size for one confirm group at a given bundle size."""
    return bundled_array_size(
        len(confirm_units_for_group(group, is_mil)), shards_per_task
    )


def resolve_confirm_bundle(
    task_index: int, group: str, is_mil: bool, shards_per_task: int
) -> list[ConfirmUnit]:
    """Return the slice of units one SLURM array task is responsible for."""
    if shards_per_task < 1:
        raise ValueError("shards per task must be positive")
    units = confirm_units_for_group(group, is_mil)
    first = task_index * shards_per_task
    return units[first : first + shards_per_task]
