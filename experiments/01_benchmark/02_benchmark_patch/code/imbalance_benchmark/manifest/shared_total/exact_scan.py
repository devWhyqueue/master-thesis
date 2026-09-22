from __future__ import annotations

import logging
import time
from collections.abc import Mapping


from imbalance_benchmark.manifest import log_every
from imbalance_benchmark.construction import (
    _allocation_is_feasible,
    allocate_counts,
    effective_rho,
)
from imbalance_benchmark.manifest.shared_total.context import (
    _Candidate,
    _FeasibilityContext,
)
from imbalance_benchmark.manifest.statistics import achieved_rho

logger = logging.getLogger(__name__)

_MODERATE_RHO = 10.0
_SEVERE_RHO = 100.0
_BALANCED_RHO = 1.0

# +/-10% of the requested ratio: the largest total whose achieved moderate
# and severe ratios both fall inside these bands, for every locked
# assignment, is preferred over the joint-maxima fallback (search.py).
_MODERATE_BAND = (9.0, 11.0)
_SEVERE_BAND = (90.0, 110.0)


def _within_band(rho: float, band: tuple[float, float]) -> bool:
    low, high = band
    return low <= rho <= high


def _counts_feasible(counts: Mapping[str, int], ctx: _FeasibilityContext) -> bool:
    """Cheap membership check against precomputed feasible counts, tightest class first."""
    return all(
        counts[name] in ctx.feasible_counts[name] for name in ctx.priority_classes
    )


def _score_total(
    ctx: _FeasibilityContext,
    balanced_available: list[int],
    assignment_availability: Mapping[str, tuple[list[str], list[int]]],
    total: int,
) -> _Candidate | None:
    """Score one total from its exact integer allocations, or reject it cheaply.

    Counts come from the same authoritative path ``_build_conditions`` uses
    downstream (``effective_rho`` then ``allocate_counts``), so a total's
    score here is exactly what the real construction would realize for it -
    not an approximation of it. ``effective_rho`` requires an unclipped
    rho=1 profile to exist before it will lower *any* condition's ratio,
    moderate and severe included; beyond that point it raises rather than
    degrading. A total this search accepted but that boundary rejects would
    blow up the real freeze, so it is checked here first, cheaply (a single
    bounds pass, no root-finding), ahead of building any allocation. Every
    other check below also fails fast rather than building allocations that
    are already known to be cap-infeasible.
    """
    if not _allocation_is_feasible(
        balanced_available, total, _BALANCED_RHO, ctx.min_support
    ):
        return None
    balanced = dict(
        zip(
            ctx.classes,
            allocate_counts(balanced_available, total, _BALANCED_RHO, ctx.min_support),
            strict=True,
        )
    )
    if not _counts_feasible(balanced, ctx):
        return None
    worst_moderate = float("inf")
    worst_severe = float("inf")
    tolerance_ok = True
    for order, available in assignment_availability.values():
        moderate_rho = effective_rho(available, _MODERATE_RHO, ctx.min_support, total)
        moderate = dict(
            zip(
                order,
                allocate_counts(available, total, moderate_rho, ctx.min_support),
                strict=True,
            )
        )
        if not _counts_feasible(moderate, ctx):
            return None
        severe_rho = effective_rho(available, _SEVERE_RHO, ctx.min_support, total)
        severe = dict(
            zip(
                order,
                allocate_counts(available, total, severe_rho, ctx.min_support),
                strict=True,
            )
        )
        if not _counts_feasible(severe, ctx):
            return None
        moderate_achieved = achieved_rho(moderate)
        severe_achieved = achieved_rho(severe)
        tolerance_ok = (
            tolerance_ok
            and _within_band(moderate_achieved, _MODERATE_BAND)
            and _within_band(severe_achieved, _SEVERE_BAND)
        )
        worst_moderate = min(worst_moderate, moderate_achieved)
        worst_severe = min(worst_severe, severe_achieved)
    return _Candidate(total, worst_moderate, worst_severe, tolerance_ok)


def _log_scan_progress(
    last_logged: float,
    total: int,
    floor: int,
    ceiling: int,
    scanned: int,
    rejected: int,
) -> float:
    return log_every(
        last_logged,
        logger,
        "freeze: shared-total scan at %d (range [%d, %d]), %d scanned, %d rejected",
        total,
        floor,
        ceiling,
        scanned,
        rejected,
    )


def _scan_candidates(
    ctx: _FeasibilityContext, floor: int, ceiling: int
) -> list[_Candidate]:
    """Exhaustively score every integer total in range; keep only compact survivors."""
    balanced_available = [ctx.supports[name] for name in ctx.classes]
    assignment_availability = {
        name: (order, [ctx.supports[name] for name in order])
        for name, order in ctx.locked_assignments.items()
    }
    start = time.perf_counter()
    last_logged = start
    rejected = 0
    candidates: list[_Candidate] = []
    for total in range(floor, ceiling + 1):
        last_logged = _log_scan_progress(
            last_logged, total, floor, ceiling, total - floor + 1, rejected
        )
        candidate = _score_total(
            ctx, balanced_available, assignment_availability, total
        )
        if candidate is None:
            rejected += 1
        else:
            candidates.append(candidate)
    logger.info(
        "freeze: shared-total scan complete: %d totals scanned, %d cheap-rejected, "
        "%d candidates, %.1fs",
        ceiling - floor + 1,
        rejected,
        len(candidates),
        time.perf_counter() - start,
    )
    return candidates
