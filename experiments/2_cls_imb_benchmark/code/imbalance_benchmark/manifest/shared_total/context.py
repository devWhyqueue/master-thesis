from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class _FeasibilityContext:
    """Everything the exact search needs except the ``total`` being scored."""

    train_df: pd.DataFrame
    is_mil: bool
    seed: int
    independent_floor: int
    min_support: int
    locked_assignments: Mapping[str, list[str]]
    supports: Mapping[str, int]
    feasible_counts: Mapping[str, set[int]]
    classes: list[str]
    priority_classes: list[str]


@dataclass(frozen=True)
class _Candidate:
    """Compact per-total scan result: no allocation maps are retained."""

    total: int
    worst_moderate: float
    worst_severe: float
