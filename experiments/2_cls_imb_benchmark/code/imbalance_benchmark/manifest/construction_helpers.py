from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from pathlib import Path

import pandas as pd

from imbalance_benchmark.common import compute_sha256
from imbalance_benchmark.construction import (
    allocate_counts,
    effective_rho,
    max_shared_total,
)
from imbalance_benchmark.manifest.sampling.patch import (
    designate_patch_pool,
    select_patches_round_robin,
)
from imbalance_benchmark.manifest.sampling.slide import select_slides_round_robin
from imbalance_benchmark.manifest.statistics import (
    natural_contribution_stats,
    support_statistics,
)
from imbalance_benchmark.manifest.statistics.selection_capacity import (
    feasible_selection_counts,
)

logger = logging.getLogger(__name__)

CONDITION_RHOS = {"balanced": 1.0, "moderate": 10.0, "severe": 100.0}


def class_support_counts(train_df: pd.DataFrame, is_mil: bool) -> dict[str, int]:
    """Count allocation units: slides for MIL and patches otherwise."""
    if is_mil:
        return train_df.groupby("cancer_type")["slide_id"].nunique().to_dict()
    return train_df["cancer_type"].value_counts().to_dict()


def class_construction_seed(seed: int, class_name: str) -> int:
    """Derive a class-identity seed independent of its assigned tail rank."""
    digest = hashlib.sha256(f"{seed}:definitive:{class_name}".encode()).hexdigest()
    return int(digest[:8], 16)


def write_natural_condition(
    train_df: pd.DataFrame, data_dir: Path, is_mil: bool
) -> dict[str, object]:
    """Write the descriptive full-training-set anchor outside controlled estimands."""
    path = data_dir / "manifest_natural.csv"
    train_df.to_csv(path, index=False)
    statistics = support_statistics(train_df)
    primary = statistics["slide" if is_mil else "patch"]
    return {
        "path": str(path),
        "sha256": compute_sha256(path),
        "note": "descriptive anchor; excluded from imbalance deficit/recovery estimands",
        "allocated_counts": primary["counts"],
        "achieved_rho": primary["achieved_rho"],
        "normalized_entropy": primary["normalized_entropy"],
        "support_statistics": statistics,
        "contribution_stats": natural_contribution_stats(train_df, is_mil),
    }


def assignment_allocations(
    train_df: pd.DataFrame,
    assignments: Mapping[str, list[str]],
    total: int,
    minimum: int,
    is_mil: bool = False,
    condition_names: tuple[str, ...] = tuple(CONDITION_RHOS),
    supports: Mapping[str, int] | None = None,
) -> dict[str, dict[str, dict[str, int]]]:
    """Allocate every condition for every locked semantic-class assignment."""
    support_counts = supports or class_support_counts(train_df, is_mil)
    return {
        assignment: {
            condition: dict(
                zip(
                    order,
                    allocate_counts(
                        [support_counts[name] for name in order],
                        total,
                        effective_rho(
                            [support_counts[name] for name in order],
                            CONDITION_RHOS[condition],
                            minimum,
                            total,
                        ),
                        minimum,
                    ),
                    strict=True,
                )
            )
            for condition in condition_names
        }
        for assignment, order in assignments.items()
    }


def designate_shared_patch_pools(
    train_df: pd.DataFrame,
    allocations: Mapping[str, Mapping[str, Mapping[str, int]]],
    independent_floor: int,
    seed: int,
) -> dict[str, pd.DataFrame]:
    """Designate one per-class pool that can realize every locked allocation."""
    required: dict[str, set[int]] = {}
    for condition_sets in allocations.values():
        for counts in condition_sets.values():
            for class_name, count in counts.items():
                required.setdefault(class_name, set()).add(count)
    return {
        class_name: designate_patch_pool(
            cast(pd.DataFrame, train_df[train_df["cancer_type"] == class_name]),
            independent_floor,
            class_construction_seed(seed, class_name),
            max(counts),
            max_pool_units=min(counts),
            required_counts=tuple(sorted(counts)),
        )
        for class_name, counts in required.items()
    }


@dataclass(frozen=True)
class _FeasibilityContext:
    """Everything a feasibility probe needs except the ``total`` being tried."""

    train_df: pd.DataFrame
    is_mil: bool
    seed: int
    independent_floor: int
    min_support: int
    locked_assignments: Mapping[str, list[str]]
    supports: Mapping[str, int]
    feasible_counts: Mapping[str, set[int]]


def _log_scan_progress(last_logged: float, total: int, start: int, floor: int) -> float:
    if time.perf_counter() - last_logged <= 30:
        return last_logged
    logger.info("freeze: shared-total scan at %d (range [%d, %d])", total, floor, start)
    return time.perf_counter()


def _total_is_feasible(ctx: _FeasibilityContext, total: int) -> bool:
    """Whether one candidate total is realizable under every selection cap."""
    allocations = assignment_allocations(
        ctx.train_df,
        ctx.locked_assignments,
        total,
        ctx.min_support,
        is_mil=ctx.is_mil,
        supports=ctx.supports,
    )
    fits = all(
        count in ctx.feasible_counts[class_name]
        for condition_sets in allocations.values()
        for counts in condition_sets.values()
        for class_name, count in counts.items()
    )
    return fits and _cap_feasible(ctx, allocations)


def cap_feasible_shared_total(
    train_df: pd.DataFrame,
    classes: list[str],
    min_support: int,
    is_mil: bool,
    seed: int,
    independent_floor: int = 10,
    assignments: Mapping[str, list[str]] | None = None,
) -> int:
    """Find the largest controlled total that satisfies the actual unit caps."""
    supports = class_support_counts(train_df, is_mil)
    floor = len(classes) * min_support
    start = max_shared_total([supports[name] for name in classes], min_support)
    ctx = _FeasibilityContext(
        train_df,
        is_mil,
        seed,
        independent_floor,
        min_support,
        assignments or {"native": classes},
        supports,
        feasible_selection_counts(train_df, min_support, is_mil),
    )
    last_logged = time.perf_counter()
    for total in range(start, floor - 1, -1):
        last_logged = _log_scan_progress(last_logged, total, start, floor)
        if _total_is_feasible(ctx, total):
            return total
    raise ValueError(
        "No shared total satisfies the independent-support and contribution caps"
    )


def _cap_feasible(
    ctx: _FeasibilityContext,
    allocations: Mapping[str, Mapping[str, Mapping[str, int]]],
) -> bool:
    """Probe every condition allocation on its designated fixed patch pool."""
    selector = select_slides_round_robin if ctx.is_mil else select_patches_round_robin
    try:
        pools = (
            designate_shared_patch_pools(
                ctx.train_df, allocations, ctx.independent_floor, ctx.seed
            )
            if not ctx.is_mil
            else {}
        )
        for condition_sets in allocations.values():
            for counts in condition_sets.values():
                for name, count in counts.items():
                    selected = selector(
                        pools.get(
                            name,
                            cast(
                                pd.DataFrame,
                                ctx.train_df[ctx.train_df["cancer_type"] == name],
                            ),
                        ),
                        count,
                        class_construction_seed(ctx.seed, name),
                    )
                    if not ctx.is_mil and not _retains_fixed_pool(
                        selected, pools[name]
                    ):
                        return False
    except ValueError:
        return False
    return True


def _retains_fixed_pool(selected: pd.DataFrame, pool: pd.DataFrame) -> bool:
    """Whether a patch condition retains every designated patient and slide."""
    return set(pool["case_id"]).issubset(selected["case_id"]) and set(
        pool["slide_id"]
    ).issubset(selected["slide_id"])
