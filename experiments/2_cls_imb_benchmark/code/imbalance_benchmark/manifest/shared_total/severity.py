from __future__ import annotations

from collections.abc import Mapping

from imbalance_benchmark.construction import effective_rho


def severity_aware_upper_bound(
    supports: Mapping[str, int],
    assignments: Mapping[str, list[str]],
    min_support: int,
    rho: float = 100.0,
) -> int:
    """Return a total wide enough for a severe (``rho``) profile's head class.

    ``max_shared_total`` deliberately ignores severity, so it can return a
    total so small that every condition floor-clips to ``min_support`` and
    the achieved ratio collapses to 1 regardless of what is requested. This
    adds back the total the most demanding locked assignment's head class
    needs to realize ``rho`` against the floor, capped by its own
    availability, so the search range is never clipped below what a real
    severity intervention requires.
    """
    if not assignments:
        raise ValueError("At least one locked assignment is required")
    k = len(next(iter(assignments.values())))
    tail_floor = (k - 1) * min_support
    return tail_floor + max(
        min(supports[order[0]], int(min_support * rho))
        for order in assignments.values()
    )


def worst_case_rho(
    supports: Mapping[str, int],
    assignments: Mapping[str, list[str]],
    min_support: int,
    total: int,
    rho: float,
) -> float:
    """The achieved ratio the least favorable locked assignment realizes.

    Whichever assignment order puts a capacity-constrained class in the head
    role is the binding one; a total that is infeasible for a given order
    (e.g. it can no longer support even a balanced split) scores 0 rather
    than raising, so the search below can move away from it.
    """
    best = float("inf")
    for order in assignments.values():
        available = [supports[name] for name in order]
        try:
            achieved = effective_rho(available, rho, min_support, total)
        except ValueError:
            return 0.0
        best = min(best, achieved)
    return best


def _severity_objective(
    supports: Mapping[str, int],
    assignments: Mapping[str, list[str]],
    min_support: int,
    total: int,
) -> float:
    """Joint objective for moderate (ρ=10) and severe (ρ=100) achieved ratios."""
    return worst_case_rho(
        supports, assignments, min_support, total, 10.0
    ) + worst_case_rho(supports, assignments, min_support, total, 100.0)


def geometric_descent(start: int, floor: int) -> list[int]:
    """Exponentially-spaced descending probe points from ``start`` down to ``floor``."""
    candidates = []
    total = start
    while total > floor:
        candidates.append(total)
        total -= max(1, total // 20)
    candidates.append(floor)
    return candidates


def severity_optimal_total(
    supports: Mapping[str, int],
    assignments: Mapping[str, list[str]],
    min_support: int,
    floor: int,
    ceiling: int,
) -> int:
    """Largest total that maximizes the worst-case achieved severity.

    Achieved severity is not monotone in the total (see ``worst_case_rho``),
    so a coarse geometric sweep locates the best region and the bracket
    around it is then scanned exhaustively for the largest tying total,
    rather than assuming the optimum sits at either end of the range.
    """
    grid = sorted(set(geometric_descent(ceiling, floor)), reverse=True)
    scored = [
        (_severity_objective(supports, assignments, min_support, total), total)
        for total in grid
    ]
    best_score, best_total = max(scored)
    peak_index = scored.index((best_score, best_total))
    bracket_high = grid[peak_index - 1] if peak_index > 0 else ceiling
    for candidate in range(bracket_high, best_total, -1):
        if (
            _severity_objective(supports, assignments, min_support, candidate)
            >= best_score
        ):
            return candidate
    return best_total
