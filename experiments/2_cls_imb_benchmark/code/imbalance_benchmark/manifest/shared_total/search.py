from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import pandas as pd

from imbalance_benchmark.construction import max_shared_total
from imbalance_benchmark.manifest.construction_helpers import (
    _retains_fixed_pool,
    assignment_allocations,
    class_construction_seed,
    class_support_counts,
    designate_shared_patch_pools,
    required_counts_by_class,
)
from imbalance_benchmark.manifest.sampling.patch import select_patches_round_robin
from imbalance_benchmark.manifest.sampling.slide import select_slides_round_robin
from imbalance_benchmark.manifest.shared_total.severity import (
    geometric_descent,
    severity_aware_upper_bound,
    severity_optimal_total,
)
from imbalance_benchmark.manifest.statistics.selection_capacity import (
    feasible_selection_counts,
)

logger = logging.getLogger(__name__)


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


def cap_feasible_shared_total(
    train_df: pd.DataFrame,
    classes: list[str],
    min_support: int,
    is_mil: bool,
    seed: int,
    independent_floor: int = 10,
    assignments: Mapping[str, list[str]] | None = None,
) -> int:
    """Largest total realizing the best joint severity every locked assignment
    can attain, subject to the actual unit caps.

    The largest total does not maximize severity: whichever assignment puts
    the scarcest class in the head role is bottlenecked by that class's own
    availability, not by ``total`` (see ``shared_total.severity``). The
    search first finds the severity-optimal total via cheap ratio checks,
    then scans downward for the largest total also surviving cap checks.
    """
    ctx, floor, ceiling = _build_search_context(
        train_df, classes, min_support, is_mil, seed, independent_floor, assignments
    )
    severity_optimal = severity_optimal_total(
        ctx.supports, ctx.locked_assignments, min_support, floor, ceiling
    )
    return _largest_feasible_total(ctx, severity_optimal, floor)


def _build_search_context(
    train_df: pd.DataFrame,
    classes: list[str],
    min_support: int,
    is_mil: bool,
    seed: int,
    independent_floor: int,
    assignments: Mapping[str, list[str]] | None,
) -> tuple[_FeasibilityContext, int, int]:
    """Everything the two-phase search needs: probe context, floor, ceiling."""
    supports = class_support_counts(train_df, is_mil)
    locked_assignments = assignments or {"native": classes}
    floor = len(classes) * min_support
    ceiling = max(
        max_shared_total([supports[name] for name in classes], min_support),
        severity_aware_upper_bound(supports, locked_assignments, min_support),
    )
    ctx = _FeasibilityContext(
        train_df,
        is_mil,
        seed,
        independent_floor,
        min_support,
        locked_assignments,
        supports,
        feasible_selection_counts(train_df, min_support, is_mil),
    )
    return ctx, floor, ceiling


def _log_scan_progress(last_logged: float, total: int, start: int, floor: int) -> float:
    if time.perf_counter() - last_logged <= 30:
        return last_logged
    logger.info("freeze: shared-total scan at %d (range [%d, %d])", total, floor, start)
    return time.perf_counter()


def _total_is_feasible(ctx: _FeasibilityContext, total: int) -> bool:
    """Whether one candidate total is realizable under every selection cap.

    The severity-aware search range can reach totals too large even for a
    balanced (ρ=1) split of the scarcest class - ``effective_rho`` raises
    rather than returning a degenerate ratio there, which simply marks the
    candidate infeasible rather than aborting the search.
    """
    try:
        allocations = assignment_allocations(
            ctx.train_df,
            ctx.locked_assignments,
            total,
            ctx.min_support,
            is_mil=ctx.is_mil,
            supports=ctx.supports,
        )
    except ValueError:
        return False
    fits = all(
        count in ctx.feasible_counts[class_name]
        for condition_sets in allocations.values()
        for counts in condition_sets.values()
        for class_name, count in counts.items()
    )
    return fits and _cap_feasible(ctx, allocations)


def _largest_feasible_total(ctx: _FeasibilityContext, start: int, floor: int) -> int:
    """Coarse-to-fine downward search for the largest cap-feasible total.

    A full per-integer scan across a range that can span tens of thousands of
    candidates re-probes an expensive pool designation at every step. A
    geometric sweep narrows to the bracket where feasibility first appears
    scanning downward, then that bracket alone is scanned exhaustively.
    Feasibility is not guaranteed monotone in the total (a contribution cap
    can reject one total while accepting a smaller one), so the bracket is
    probed in full rather than bisected.
    """
    last_logged = time.perf_counter()
    previous = start
    for candidate in geometric_descent(start, floor):
        last_logged = _log_scan_progress(last_logged, candidate, start, floor)
        if _total_is_feasible(ctx, candidate):
            for total in range(previous, candidate - 1, -1):
                if _total_is_feasible(ctx, total):
                    return total
        previous = candidate - 1
    raise ValueError(
        "No shared total satisfies the independent-support and contribution caps"
    )


def _cap_feasible(
    ctx: _FeasibilityContext,
    allocations: Mapping[str, Mapping[str, Mapping[str, int]]],
) -> bool:
    """Probe every condition allocation on its designated fixed patch pool.

    Only the largest count assigned to a class must retain (cover) the whole
    designated pool; a smaller tail-condition count only needs to draw from it.
    """
    selector = select_slides_round_robin if ctx.is_mil else select_patches_round_robin
    max_required = {
        class_name: max(counts)
        for class_name, counts in required_counts_by_class(allocations).items()
    }
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
                    if (
                        not ctx.is_mil
                        and count == max_required[name]
                        and not _retains_fixed_pool(selected, pools[name])
                    ):
                        return False
    except ValueError:
        return False
    return True
