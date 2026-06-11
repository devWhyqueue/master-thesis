from typing import cast

import numpy as np
import pandas as pd


def power_law_counts(
    available: pd.Series,
    ordered_classes: list[str],
    parameter: float,
    total: int,
) -> dict[str, int]:
    """Return feasible full-size power-law target counts."""
    if len(ordered_classes) != len(available):
        raise ValueError("Class order must contain every available class exactly once.")
    minimum = pd.Series(1, index=ordered_classes, dtype=int)
    capacity = available.reindex(ordered_classes).astype(int)
    remaining = total - int(minimum.sum())
    if remaining < 0:
        raise ValueError("Training split is too small to keep every class represented.")
    extra = _redistributed_extra_counts(capacity - minimum, parameter, remaining)
    return (minimum + extra).astype(int).to_dict()


def target_counts(
    available: pd.Series,
    ordered_classes: list[str],
    parameter: float,
    total: int,
    overflow_strategy: str,
) -> dict[str, int]:
    """Return target class counts for the requested overflow strategy."""
    if overflow_strategy == "redistribute":
        return power_law_counts(available, ordered_classes, parameter, total)
    if overflow_strategy != "replacement":
        raise ValueError(f"Unknown overflow strategy: {overflow_strategy}")
    return replacement_counts(ordered_classes, parameter, total)


def replacement_counts(
    ordered_classes: list[str],
    parameter: float,
    total: int,
) -> dict[str, int]:
    """Return power-law counts that permit replacement above class capacity."""
    remaining = total - len(ordered_classes)
    raw = _raw_power_law_targets(parameter, remaining, len(ordered_classes))
    extra = _integer_targets(raw, remaining)
    minimum = pd.Series(1, index=ordered_classes, dtype=int)
    return (minimum + pd.Series(extra, index=ordered_classes)).astype(int).to_dict()


def _redistributed_extra_counts(
    capacities: pd.Series,
    parameter: float,
    remaining: int,
) -> pd.Series:
    extra = pd.Series(0, index=capacities.index, dtype=int)
    active = cast(pd.Series, capacities[capacities > 0].copy())
    while remaining > 0 and not active.empty:
        raw = _raw_power_law_targets(parameter, remaining, len(active))
        allocation = pd.Series(_integer_targets(raw, remaining), index=active.index)
        capped = allocation > active
        if not bool(capped.any()):
            extra.loc[active.index] += allocation.astype(int)
            return extra
        capped_active = cast(pd.Series, active[capped])
        extra.loc[capped_active.index] += capped_active.astype(int)
        remaining -= int(capped_active.sum())
        active = cast(pd.Series, active[~capped])
    if remaining != 0:
        raise ValueError("Could not redistribute all requested slides.")
    return extra


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
