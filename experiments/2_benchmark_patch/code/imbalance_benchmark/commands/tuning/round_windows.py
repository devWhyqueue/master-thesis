from __future__ import annotations

from pathlib import Path

from imbalance_benchmark.modeling.context import (
    GRIDS,
    LEARNING_RATE_GRID,
    NO_STRENGTH_GRID_METHODS,
)
from imbalance_benchmark.modeling.workflows.tuning.aggregation.candidate_registry import (
    load_round_grids,
)

__all__ = ["Window", "round0_windows", "this_round_windows"]

Window = tuple[list[float], list[float] | None]


def round0_windows(methods: tuple[str, ...], n_classes: int) -> dict[str, Window]:
    """Every method's strength window, OKO's k capped at n_classes - 1."""
    windows = {}
    for method in methods:
        if method in NO_STRENGTH_GRID_METHODS:
            windows[method] = (LEARNING_RATE_GRID, None)
            continue
        values = [float(v) for v in GRIDS[method]]
        if method == "oko":
            values = sorted({v for v in values if int(v) <= n_classes - 1})
        windows[method] = (LEARNING_RATE_GRID, values)
    return windows


def this_round_windows(
    root: Path,
    condition: str,
    phase: str,
    round_index: int,
    methods: tuple[str, ...],
    n_classes: int,
) -> dict[str, Window]:
    """Round 0 uses the frozen defaults; a later round reads its signed active windows."""
    if round_index == 0:
        return round0_windows(methods, n_classes)
    round_grids = load_round_grids(root, condition, phase)
    return {
        method: (window["lr_window"], window.get("strength_window"))
        for method, window in round_grids["windows"].items()
        if method in methods
    }
