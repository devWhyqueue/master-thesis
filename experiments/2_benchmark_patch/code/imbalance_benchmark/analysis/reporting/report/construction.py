from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from imbalance_benchmark.analysis.reporting.report.sources import (
    ASSIGNMENT,
    CONDITION,
    Dataset,
    body,
    num,
)

__all__ = ["realized_support", "support_frame"]

_TOLERANCE = {"moderate": (9.0, 11.0), "severe": (90.0, 110.0)}
# balanced_spread's nominal rho is pinned to 1.0 by construction; severe_spread
# shares severe's nominal target.
_UNCHECKED = {"natural", "balanced", "balanced_spread"}
_DEGENERATE = 1.05


def _conditions(freeze: dict[str, Any]) -> Iterator[tuple[str, str, dict[str, Any]]]:
    """Every realized training set in one split: anchor, balanced, and controlled."""
    yield "natural", "unassigned", freeze["natural"]
    yield "balanced", "unassigned", freeze["conditions"]["balanced"]
    for assignment, conditions in sorted(freeze["assignment_conditions"].items()):
        for severity, entry in sorted(conditions.items()):
            yield severity, assignment, entry


def _independent_counts(entry: dict[str, Any]) -> tuple[int, int]:
    """Unique partitioning units and slides in one realized training set.

    A unit can supply more than one class, so per-class counts cannot simply be
    summed; the manifest is the only place the union is recorded.
    """
    frame = pd.read_csv(entry["path"], usecols=["case_id", "slide_id"])
    return int(frame["case_id"].nunique()), int(frame["slide_id"].nunique())


def _status(severity: str, achieved: float) -> str:
    if severity in _UNCHECKED:
        return "---"
    if achieved <= _DEGENERATE:
        return "Degenerate"
    low, high = _TOLERANCE[severity.removesuffix("_spread")]
    return "On target" if low <= achieved <= high else "Off target"


def support_frame(datasets: list[Dataset]) -> pd.DataFrame:
    """Realized training support for every dataset, split, and condition."""
    rows = []
    for data in datasets:
        for split, freeze in enumerate(data.freezes):
            for severity, assignment, entry in _conditions(freeze):
                units, slides = _independent_counts(entry)
                patches = sum(entry["support_statistics"]["patch"]["counts"].values())
                rows.append(
                    {
                        "dataset": data.name,
                        "split": split,
                        "condition": severity,
                        "assignment": assignment,
                        "units": units,
                        "slides": slides,
                        "patches": patches,
                        "achieved_rho": entry["achieved_rho"],
                        "normalized_entropy_deficit": entry["normalized_entropy"],
                        "status": _status(severity, entry["achieved_rho"]),
                    }
                )
    return pd.DataFrame(rows)


def realized_support(datasets: list[Dataset], csv_path: Path | None = None) -> str:
    """Realized construction per dataset, split, and condition, with tolerance flags."""
    frame = support_frame(datasets)
    if csv_path is not None:
        frame.to_csv(csv_path, index=False)
    rendered = pd.DataFrame(
        {
            "Dataset": frame["dataset"],
            "Split": frame["split"],
            "Condition": frame["condition"].map(CONDITION),
            "Assignment": frame["assignment"].map(ASSIGNMENT),
            "Units": frame["units"],
            "Slides": frame["slides"],
            "Patches": frame["patches"],
            r"$\rho$": [num(value, 3) for value in frame["achieved_rho"]],
            r"$1-H_{\mathrm{norm}}$": [
                num(value, 4) for value in frame["normalized_entropy_deficit"]
            ],
            "Tolerance": frame["status"],
        }
    )
    return body(rendered, longtable=True)
