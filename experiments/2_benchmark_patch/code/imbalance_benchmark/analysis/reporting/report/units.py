"""Comparison units: their reading order, and their rows in an endpoint table."""

from __future__ import annotations

from typing import cast

import pandas as pd

from imbalance_benchmark.analysis.reporting.report.sources import Dataset

__all__ = ["comparison_units", "endpoint_cell"]

# Reading order for the crossed condition family: the independent-support
# contrast first (balanced against its spread reference), then the nominal-rho
# ladder, then the same rho re-run over the spread pool. The spread references
# are themselves never comparison units.
_SEVERITY_ORDER = {"balanced": 0, "moderate": 1, "severe": 2, "severe_spread": 3}
_ASSIGNMENT_ORDER = {"native": 0, "difficulty_aligned": 1, "difficulty_reversed": 2}
# The balanced allocation and the natural anchor induce no tail ordering, so the
# endpoint tables file them once under ``unassigned`` even though a unit using
# one as its deprived arm still carries a real tail assignment.
_UNASSIGNED_CONDITIONS = frozenset({"balanced", "natural"})


def comparison_units(data: Dataset) -> list[tuple[str, str]]:
    """Every (tail assignment, severity) comparison unit this dataset realized."""
    units = {
        (row["assignment"], row["severity"])
        for row in data.comparisons
        if row["method"] == "ce"
    }
    return sorted(units, key=lambda u: (_SEVERITY_ORDER[u[1]], _ASSIGNMENT_ORDER[u[0]]))


def endpoint_cell(table: pd.DataFrame, unit: tuple[str, str]) -> pd.DataFrame:
    """Rows of an endpoint table belonging to one unit's deprived arm."""
    assignment = "unassigned" if unit[1] in _UNASSIGNED_CONDITIONS else unit[0]
    mask = (table["assignment"] == assignment) & (table["condition"] == unit[1])
    return cast(pd.DataFrame, table[mask])
