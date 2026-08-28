from __future__ import annotations

from typing import cast

import pandas as pd

from imbalance_benchmark.analysis.inference.confirmatory.holm import PRIMARY_METHODS
from imbalance_benchmark.analysis.reporting.report.sources import (
    ASSIGNMENT,
    CONDITION,
    METHOD,
    Dataset,
    body,
    ci,
    num,
)

__all__ = ["calibration_detail", "classwise_endpoints", "cost"]

_FAMILY = sorted(PRIMARY_METHODS)
_WITH_REFERENCE = sorted(PRIMARY_METHODS | {"ce"})
_PROBABILITY = {
    "negative_log_likelihood": "NLL",
    "brier_score": "Brier",
    "expected_calibration_error": "ECE",
    "temperature_scaled_negative_log_likelihood": "Temp.\\ NLL",
    "temperature_scaled_brier_score": "Temp.\\ Brier",
    "temperature_scaled_expected_calibration_error": "Temp.\\ ECE",
}
_COST = {
    "processed_examples": "Processed examples",
    "unique_examples_exposed": "Unique exposures",
    "accelerator_hours": "Accelerator hours",
    "peak_accelerator_memory_bytes": "Peak memory (bytes)",
    "total_parameters": "Parameters",
}
_INTERVAL_ENDPOINTS = ("expected_calibration_error",)


def _labelled(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    """Prefix one dataset's endpoint table with report-facing labels."""
    labelled = frame.copy()
    labelled.insert(0, "Dataset", name)
    labelled["assignment"] = labelled["assignment"].map(ASSIGNMENT.get)
    labelled["condition"] = labelled["condition"].map(CONDITION.get)
    labelled["method"] = labelled["method"].map(lambda m: METHOD.get(m, m))
    return labelled


def classwise_endpoints(datasets: list[Dataset]) -> str:
    """Per-class CE endpoints with the tier each condition assigns them."""
    frames = []
    for data in datasets:
        frame = data.tables["equal_split_classwise_endpoints"]
        rows = cast(pd.DataFrame, frame[frame["method"] == "ce"])
        frames.append(_labelled(rows, data.name))
    combined = pd.concat(frames, ignore_index=True).drop(columns=["method"])
    combined["class_name"] = combined["class_name"].str.replace("_", " ")
    combined.columns = pd.Index(
        [
            "Dataset",
            "Assignment",
            "Condition",
            "Class",
            "Tier",
            "Recall",
            "$F_1$",
            "NLL",
            "Brier",
        ]
    )
    return body(combined, longtable=True)


def _probability_cell(rows: pd.DataFrame, endpoint: str) -> str:
    match = rows[rows["endpoint"] == endpoint]
    if match.empty:
        return "---"
    entry = match.iloc[0]
    value = num(entry["estimate"])
    if endpoint not in _INTERVAL_ENDPOINTS:
        return value
    return f"{value} {ci((entry['ci_low'], entry['ci_high']))}"


def calibration_detail(datasets: list[Dataset]) -> str:
    """Raw and temperature-scaled probability quality per condition and method."""
    rows = []
    for data in datasets:
        table = data.tables["secondary_endpoint_intervals"]
        family = cast(pd.DataFrame, table[table["method"].isin(_WITH_REFERENCE)])
        for key, group in family.groupby(
            ["assignment", "condition", "method"], sort=True
        ):
            assignment, condition, method = cast(tuple[str, str, str], key)
            rows.append(
                {
                    "Dataset": data.name,
                    "Assignment": ASSIGNMENT[assignment],
                    "Condition": CONDITION[condition],
                    "Method": METHOD.get(method, method),
                    **{
                        label: _probability_cell(group, endpoint)
                        for endpoint, label in _PROBABILITY.items()
                    },
                }
            )
    return body(pd.DataFrame(rows), longtable=True)


def _significant(value: float) -> str:
    """Cost cell across scales, without rendering a rounded zero as ``-0``."""
    return "0" if abs(value) < 5e-4 else f"{value:.4g}"


def _cost_frame(datasets: list[Dataset]) -> pd.DataFrame:
    frames = []
    for data in datasets:
        table = data.tables["cost_comparison_intervals"]
        selected = cast(
            pd.DataFrame,
            table[table["method"].isin(_FAMILY) & table["endpoint"].isin(list(_COST))],
        )
        frames.append(_labelled(selected, data.name))
    return pd.concat(frames, ignore_index=True)


def cost(datasets: list[Dataset]) -> str:
    """Matched cost effects against CE in the same condition, five-family only."""
    combined = _cost_frame(datasets)
    rendered = pd.DataFrame(
        {
            "Dataset": combined["Dataset"],
            "Assignment": combined["assignment"],
            "Condition": combined["condition"],
            "Method": combined["method"],
            "Endpoint": combined["endpoint"].map(_COST.get),
            "Effect": [_significant(value) for value in combined["effect"]],
            r"95\% CI": [
                f"[{_significant(low)}, {_significant(high)}]"
                for low, high in zip(
                    combined["ci_low"], combined["ci_high"], strict=True
                )
            ],
        }
    )
    return body(rendered, longtable=True)
