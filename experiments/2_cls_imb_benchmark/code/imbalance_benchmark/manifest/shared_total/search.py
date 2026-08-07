from __future__ import annotations

import logging
import time
from collections.abc import Mapping
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
from imbalance_benchmark.manifest.shared_total.context import (
    _Candidate,
    _FeasibilityContext,
)
from imbalance_benchmark.manifest.shared_total.exact_scan import _scan_candidates
from imbalance_benchmark.manifest.shared_total.severity import (
    severity_aware_upper_bound,
)
from imbalance_benchmark.manifest.statistics.selection_capacity import (
    feasible_selection_counts,
)

logger = logging.getLogger(__name__)


def cap_feasible_shared_total(
    train_df: pd.DataFrame,
    classes: list[str],
    min_support: int,
    is_mil: bool,
    seed: int,
    independent_floor: int = 10,
    assignments: Mapping[str, list[str]] | None = None,
) -> int:
    """Largest total within tolerance of the requested severities, subject to
    the actual unit caps; falls back to the best jointly attainable total.

    Every integer total in ``[floor, ceiling]`` is scored exactly from its
    integer allocation and cheaply rejected against the precomputed
    contribution caps. Among the survivors, the largest total whose moderate
    and severe ratios both fall within tolerance of the requested severities
    for every locked assignment is preferred, confirmed against the true
    fixed-pool selection - the only place an expensive probe runs. A
    dataset-regime with no such total (or where every one fails the probe)
    falls back to the total attaining the best jointly attainable severity.
    """
    ctx, floor, ceiling = _build_search_context(
        train_df, classes, min_support, is_mil, seed, independent_floor, assignments
    )
    candidates = _scan_candidates(ctx, floor, ceiling)
    if not candidates:
        raise ValueError(
            "No shared total satisfies the independent-support and contribution caps"
        )
    tolerance_total = _largest_tolerance_feasible_total(ctx, candidates)
    if tolerance_total is not None:
        return tolerance_total
    return _largest_jointly_optimal_total(ctx, candidates)


def _largest_tolerance_feasible_total(
    ctx: _FeasibilityContext, candidates: list[_Candidate]
) -> int | None:
    """Largest cap-feasible total within tolerance of both requested severities.

    Tried largest-first against the true fixed-pool construction; a probe
    rejection excludes only that total and falls through to the next, never
    substituting a total the tolerance band did not already accept and never
    falling back silently to a compromise the caller cannot see.
    """
    for total in sorted((c.total for c in candidates if c.tolerance_ok), reverse=True):
        if _total_cap_feasible(ctx, total):
            return total
    return None


def _build_search_context(
    train_df: pd.DataFrame,
    classes: list[str],
    min_support: int,
    is_mil: bool,
    seed: int,
    independent_floor: int,
    assignments: Mapping[str, list[str]] | None,
) -> tuple[_FeasibilityContext, int, int]:
    """Everything the exact search needs: probe context, floor, ceiling."""
    supports = class_support_counts(train_df, is_mil)
    locked_assignments = assignments or {"native": classes}
    floor = len(classes) * min_support
    ceiling = max(
        max_shared_total([supports[name] for name in classes], min_support),
        severity_aware_upper_bound(supports, locked_assignments, min_support),
    )
    feasible_counts = feasible_selection_counts(train_df, min_support, is_mil)
    ctx = _FeasibilityContext(
        train_df,
        is_mil,
        seed,
        independent_floor,
        min_support,
        locked_assignments,
        supports,
        feasible_counts,
        classes,
        sorted(classes, key=lambda name: len(feasible_counts[name])),
    )
    return ctx, floor, ceiling


# --- joint-maximum selection and pool confirmation -----------------------


def _largest_jointly_optimal_total(
    ctx: _FeasibilityContext, candidates: list[_Candidate]
) -> int:
    """The largest total attaining both global maxima that also clears the pool probe.

    A total tying both the moderate and severe maxima can still fail the
    true fixed-pool selection (a contribution cap is not monotone in the
    total). When every currently-tied total fails, those totals are
    excluded and the maxima are recomputed over what remains, rather than
    settling for a total that only ties one of the two ratios.
    """
    start = time.perf_counter()
    probes = 0
    excluded: set[int] = set()
    while True:
        remaining = [
            candidate for candidate in candidates if candidate.total not in excluded
        ]
        if not remaining:
            raise ValueError(
                "No shared total satisfies the independent-support and contribution caps"
            )
        best_moderate = max(candidate.worst_moderate for candidate in remaining)
        best_severe = max(candidate.worst_severe for candidate in remaining)
        joint = sorted(
            (
                candidate
                for candidate in remaining
                if candidate.worst_moderate == best_moderate
                and candidate.worst_severe == best_severe
            ),
            key=lambda candidate: candidate.total,
            reverse=True,
        )
        if not joint:
            raise ValueError(
                "No shared total attains the moderate and severe rho maxima "
                "simultaneously across every locked assignment"
            )
        for candidate in joint:
            probes += 1
            if _total_cap_feasible(ctx, candidate.total):
                logger.info(
                    "freeze: shared-total pool probes: %d, %.1fs",
                    probes,
                    time.perf_counter() - start,
                )
                return candidate.total
            excluded.add(candidate.total)


def _total_cap_feasible(ctx: _FeasibilityContext, total: int) -> bool:
    """Whether one total is realizable under every cap via the true fixed-pool
    construction - the same path ``_build_conditions`` uses downstream."""
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
    return _cap_feasible(ctx, allocations)


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
