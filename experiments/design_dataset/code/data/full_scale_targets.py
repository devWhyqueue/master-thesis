from typing import cast

import numpy as np
import pandas as pd


def power_law_counts(
    available: pd.Series,
    ordered_classes: list[str],
    parameter: float,
    total: int,
) -> dict[str, int]:
    """Return strict full-size power-law target counts."""
    if len(ordered_classes) != len(available):
        raise ValueError("Class order must contain every available class exactly once.")
    minimum = pd.Series(1, index=ordered_classes, dtype=int)
    capacity = available.reindex(ordered_classes).astype(int)
    remaining = total - int(minimum.sum())
    if remaining < 0:
        raise ValueError("Training split is too small to keep every class represented.")
    extra = _extra_counts(ordered_classes, parameter, remaining)
    counts = minimum + extra
    _raise_if_infeasible(counts, capacity, parameter, total)
    return counts.astype(int).to_dict()


def is_feasible_total(
    available: pd.Series,
    ordered_classes: list[str],
    parameter: float,
    total: int,
) -> bool:
    """Return whether one strict power-law target total is feasible."""
    try:
        power_law_counts(available, ordered_classes, parameter, total)
    except ValueError:
        return False
    return True


def max_feasible_total(
    available: pd.Series,
    ordered_classes: list[str],
    parameter: float,
) -> int:
    """Return the largest strict total that stays within class capacities."""
    lower = len(ordered_classes)
    upper = int(available.sum())
    best = lower
    while lower <= upper:
        midpoint = (lower + upper) // 2
        if is_feasible_total(available, ordered_classes, parameter, midpoint):
            best = midpoint
            lower = midpoint + 1
            continue
        upper = midpoint - 1
    return best


def _extra_counts(
    ordered_classes: list[str], parameter: float, remaining: int
) -> pd.Series:
    return pd.Series(
        _integer_targets(
            _raw_power_law_targets(parameter, remaining, len(ordered_classes)),
            remaining,
        ),
        index=ordered_classes,
        dtype=int,
    )


def _raise_if_infeasible(
    counts: pd.Series,
    capacity: pd.Series,
    parameter: float,
    total: int,
) -> None:
    violations = cast(pd.Series, counts[counts > capacity])
    if violations.empty:
        return
    class_name = str(violations.index[0])
    requested = int(violations.iloc[0])
    maximum = int(capacity.loc[class_name])
    raise ValueError(
        "Infeasible power-law target: "
        f"class={class_name}, requested={requested}, available={maximum}, "
        f"lambda={parameter}, total={total}"
    )


def _raw_power_law_targets(
    parameter: float,
    total: int,
    n_classes: int,
) -> np.ndarray:
    ranks = np.arange(1, n_classes + 1, dtype=np.float64)
    weights = np.power(ranks, -parameter)
    return weights / weights.sum() * total


def _integer_targets(raw: np.ndarray, total: int) -> np.ndarray:
    counts = np.floor(raw).astype(int)
    missing = total - int(counts.sum())
    remainders = raw - counts
    for index in np.argsort(remainders)[::-1][:missing]:
        counts[index] += 1
    return counts
