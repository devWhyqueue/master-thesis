from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Callable, cast

from pathlib import Path

import pandas as pd

from imbalance_benchmark.common import compute_sha256
from imbalance_benchmark.construction import (
    allocate_counts,
    effective_rho,
    max_shared_total,
)
from imbalance_benchmark.manifest.construction_sampling import (
    designate_patch_pool,
    select_patches_round_robin,
    select_slides_round_robin,
)
from imbalance_benchmark.manifest.statistics import (
    natural_contribution_stats,
    support_statistics,
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
) -> dict[str, dict[str, dict[str, int]]]:
    """Allocate every condition for every locked semantic-class assignment."""
    supports = class_support_counts(train_df, is_mil)
    return {
        assignment: {
            condition: dict(
                zip(
                    order,
                    allocate_counts(
                        [supports[name] for name in order],
                        total,
                        effective_rho(
                            [supports[name] for name in order],
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
    maximums: dict[str, int] = {}
    for condition_sets in allocations.values():
        for counts in condition_sets.values():
            for class_name, count in counts.items():
                maximums[class_name] = max(maximums.get(class_name, 0), count)
    return {
        class_name: designate_patch_pool(
            cast(pd.DataFrame, train_df[train_df["cancer_type"] == class_name]),
            independent_floor,
            class_construction_seed(seed, class_name),
            maximum,
        )
        for class_name, maximum in maximums.items()
    }


def _log_search_progress(last_logged: float, lo: int, hi: int) -> float:
    if time.perf_counter() - last_logged <= 30:
        return last_logged
    logger.info("freeze: shared-total search narrowed to [%d, %d]", lo, hi)
    return time.perf_counter()


@dataclass(frozen=True)
class _FeasibilityContext:
    """Everything a feasibility probe needs except the ``total`` being tried."""

    train_df: pd.DataFrame
    min_support: int
    selector: Callable[..., pd.DataFrame]
    is_mil: bool
    seed: int
    independent_floor: int


def _binary_search_feasible_total(
    ctx: _FeasibilityContext,
    assignments: Mapping[str, list[str]],
    floor: int,
    start: int,
) -> int:
    """Largest total in ``(floor, start]`` where ``_cap_feasible`` holds.

    Assumes feasibility is monotonic: every condition's per-class allocation
    and its cap shrink together as total shrinks, so feasibility only gets
    easier below any total that already works.
    """
    lo, hi, last_logged = floor, start, time.perf_counter()
    while hi - lo > 1:
        mid = (lo + hi) // 2
        ok = _cap_feasible(ctx, assignments, mid)
        lo, hi = (mid, hi) if ok else (lo, mid)
        last_logged = _log_search_progress(last_logged, lo, hi)
    return lo


def cap_feasible_shared_total(
    train_df: pd.DataFrame,
    classes: list[str],
    min_support: int,
    is_mil: bool,
    seed: int,
    independent_floor: int = 10,
    assignments: Mapping[str, list[str]] | None = None,
) -> int:
    """Find the largest controlled total that satisfies the actual unit caps.

    A linear scan here is O(range) with an expensive per-step check and can
    take hours on a large patch pool, so this binary searches instead.
    """
    supports = class_support_counts(train_df, is_mil)
    available = [supports[name] for name in classes]
    selector = select_slides_round_robin if is_mil else select_patches_round_robin
    locked_assignments = assignments or {"native": classes}
    ctx = _FeasibilityContext(
        train_df, min_support, selector, is_mil, seed, independent_floor
    )
    floor, start = len(classes) * min_support, max_shared_total(available, min_support)
    if not _cap_feasible(ctx, locked_assignments, floor):
        raise ValueError(
            "No shared total satisfies the independent-support and contribution caps"
        )
    if _cap_feasible(ctx, locked_assignments, start):
        return start
    logger.info("freeze: shared-total binary search between %d and %d", floor, start)
    return _binary_search_feasible_total(ctx, locked_assignments, floor, start)


def _cap_feasible(
    ctx: _FeasibilityContext, assignments: Mapping[str, list[str]], total: int
) -> bool:
    """Probe every condition allocation on its designated fixed patch pool."""
    try:
        allocations = assignment_allocations(
            ctx.train_df, assignments, total, ctx.min_support, is_mil=ctx.is_mil
        )
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
                    selected = ctx.selector(
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
    """Whether a condition's selection is drawn entirely from its designated pool.

    ``pool`` is sized to the largest count any condition needs for this class,
    so most conditions select a strict subset of it. Checking the reverse
    (every pool patient/slide present in the selection) can only ever hold
    when a condition's count equals that maximum -- i.e. almost never.
    """
    return set(selected["case_id"]).issubset(pool["case_id"]) and set(
        selected["slide_id"]
    ).issubset(pool["slide_id"])
