from __future__ import annotations

from collections.abc import Mapping


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
