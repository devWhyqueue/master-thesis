from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from imbalance_benchmark.manifest.construction_helpers import (
    designate_narrowed_patch_pools,
)

BuildConditions = Callable[..., dict[str, Any]]

# Plan 03's measured decision (measurement-20260829-r2.json, sha256=3a80d128ae
# 2448d88751c0316c98a2174f41419920e251d8857cda52fac6471a): which assignments
# carry a cap-feasible narrowed arm per dataset. Absent/() = nominal arm only.
NARROWED_ASSIGNMENTS_BY_DATASET: dict[str, tuple[str, ...] | None] = {
    "bracs": None,  # None = every locked assignment
    "camelyon16": ("native",),
    "tcga_ut": (),
    "panda": (),
}


def narrowed_assignments_for_dataset(
    dataset_name: str, assignments: dict[str, list[str]]
) -> tuple[str, ...]:
    """Assignments that carry a narrowed independent-support arm for this dataset."""
    scoped = NARROWED_ASSIGNMENTS_BY_DATASET.get(dataset_name, ())
    return tuple(assignments) if scoped is None else scoped


def _narrowed_classes(full_allocations: dict[str, Any], assignment: str) -> list[str]:
    """Classes the paired severe cell deprives nominally ("Which classes to narrow").

    Makes ``balanced_narrow`` the pure-independent analogue of ``severe``:
    same damaged class set, different axis.
    """
    balanced = full_allocations[assignment]["balanced"]
    severe = full_allocations[assignment]["severe"]
    return sorted(name for name, count in severe.items() if count < balanced[name])


@dataclass(frozen=True)
class NarrowingContext:
    """Shared construction inputs for one split's narrowed-arm addition."""

    train_df: pd.DataFrame
    assignments: dict[str, list[str]]
    full_allocations: dict[str, Any]
    shared_t: int
    min_support: int
    construction_seed: int
    data_dir: Path
    independent_floor: int
    shared_pools: dict[str, pd.DataFrame]
    max_required: dict[str, int] | None


def _add_one_assignment(
    build_conditions: BuildConditions,
    ctx: NarrowingContext,
    assignment: str,
    classes: list[str],
) -> dict[str, Any]:
    """Designate one assignment's narrowed pools and build its narrow conditions."""
    narrow_pools, narrow_ratio = designate_narrowed_patch_pools(
        ctx.train_df,
        ctx.full_allocations,
        ctx.independent_floor,
        ctx.construction_seed,
        set(classes),
        ctx.shared_pools,
    )
    return build_conditions(
        ctx.train_df,
        ctx.assignments[assignment],
        ctx.shared_t,
        ctx.min_support,
        False,
        ctx.construction_seed,
        ctx.data_dir,
        file_prefix=f"{assignment}_",
        condition_names=("balanced_narrow", "severe_narrow"),
        independent_floor=ctx.independent_floor,
        fixed_pools={**ctx.shared_pools, **narrow_pools},
        max_required_counts=ctx.max_required,
        narrowed_classes=classes,
        narrowed_ratio=narrow_ratio,
    )


def add_narrowed_conditions(
    build_conditions: BuildConditions,
    assignment_conditions: dict[str, dict[str, Any]],
    ctx: NarrowingContext,
    dataset_name: str,
) -> None:
    """Add balanced_narrow/severe_narrow beside moderate/severe, where scoped.

    Plan 03's measured decision scopes which assignments get a cap-feasible
    narrowed arm per dataset; other assignments keep the nominal arm only.
    ``build_conditions`` is injected (rather than imported) to avoid a
    freezing.py <-> shared_total.narrowing import cycle.
    """
    for assignment in narrowed_assignments_for_dataset(dataset_name, ctx.assignments):
        classes = _narrowed_classes(ctx.full_allocations, assignment)
        if not classes:
            continue
        narrow_conditions = _add_one_assignment(
            build_conditions, ctx, assignment, classes
        )
        assignment_conditions[assignment] = {
            **assignment_conditions[assignment],
            **narrow_conditions,
        }
