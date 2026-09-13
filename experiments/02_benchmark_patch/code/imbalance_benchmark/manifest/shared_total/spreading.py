from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from imbalance_benchmark.manifest.construction_helpers import (
    designate_spread_patch_pools,
)
from imbalance_benchmark.manifest.shared_total.degenerate import reject_non_nested_pools

BuildConditions = Callable[..., dict[str, Any]]

# Cost knob for later datasets. None means every locked assignment.
SPREAD_ASSIGNMENTS_BY_DATASET: dict[str, tuple[str, ...] | None] = {
    "bracs": None,
    "camelyon16": None,
    "tcga_ut": None,
    "panda": None,
}


def spread_assignments_for_dataset(
    dataset_name: str, assignments: dict[str, list[str]]
) -> tuple[str, ...]:
    """Return assignments carrying the spread independent-support arm."""
    scoped = SPREAD_ASSIGNMENTS_BY_DATASET.get(dataset_name)
    return tuple(assignments) if scoped is None else scoped


@dataclass(frozen=True)
class SpreadingContext:
    """Shared construction inputs for one split's spread-arm addition."""

    train_df: pd.DataFrame
    assignments: dict[str, list[str]]
    full_allocations: dict[str, Any]
    shared_t: int
    min_support: int
    construction_seed: int
    data_dir: Path
    independent_floor: int
    concentrated_pools: dict[str, pd.DataFrame]
    max_required: dict[str, int] | None


def _spread_tail_classes(ratio: dict[str, float]) -> list[str]:
    count = math.ceil(len(ratio) / 3)
    return sorted(ratio, key=lambda name: (-math.log(ratio[name]), name))[:count]


def _add_one_assignment(
    build_conditions: BuildConditions, ctx: SpreadingContext, assignment: str
) -> dict[str, Any]:
    """Designate one assignment's spread pools and build its spread conditions."""
    spread_pools, spread_ratio = designate_spread_patch_pools(
        ctx.train_df,
        ctx.full_allocations,
        ctx.independent_floor,
        ctx.construction_seed,
        ctx.concentrated_pools,
    )
    reject_non_nested_pools(ctx.concentrated_pools, spread_pools)
    return build_conditions(
        ctx.train_df,
        ctx.assignments[assignment],
        ctx.shared_t,
        ctx.min_support,
        False,
        ctx.construction_seed,
        ctx.data_dir,
        file_prefix=f"{assignment}_",
        condition_names=("balanced_spread", "severe_spread"),
        independent_floor=ctx.independent_floor,
        fixed_pools=spread_pools,
        max_required_counts=ctx.max_required,
        spread_classes=sorted(spread_pools),
        spread_ratio=spread_ratio,
        spread_tail_classes=_spread_tail_classes(spread_ratio),
    )


def add_spread_conditions(
    build_conditions: BuildConditions,
    assignment_conditions: dict[str, dict[str, Any]],
    ctx: SpreadingContext,
    dataset_name: str,
) -> None:
    """Add balanced_spread/severe_spread beside concentrated conditions."""
    for assignment in spread_assignments_for_dataset(dataset_name, ctx.assignments):
        assignment_conditions[assignment] = {
            **assignment_conditions[assignment],
            **_add_one_assignment(build_conditions, ctx, assignment),
        }


def add_concentrated_conditions(
    assignment_conditions: dict[str, dict[str, Any]], native_balanced: dict[str, Any]
) -> None:
    """Expose the reusable concentrated balanced cell in every assignment."""
    for assignment, conditions in assignment_conditions.items():
        tail = conditions.get("balanced_spread", {}).get("spread_tail_classes")
        assignment_conditions[assignment] = {
            "balanced": {**native_balanced, "spread_tail_classes": tail},
            **conditions,
        }
