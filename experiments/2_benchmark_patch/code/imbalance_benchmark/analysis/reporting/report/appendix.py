from __future__ import annotations

from typing import Any

import pandas as pd

from imbalance_benchmark.analysis.query import EXPECTED_CONFIRMATION_SEEDS
from imbalance_benchmark.analysis.reporting.report.rq1 import ce_row
from imbalance_benchmark.analysis.reporting.report.units import comparison_units
from imbalance_benchmark.analysis.reporting.report.sources import (
    ASSIGNMENT,
    CONDITION,
    METHOD,
    Dataset,
    body,
    num,
)

__all__ = [
    "completeness",
    "method_diagnostics",
    "per_split",
    "preflight_outcome",
    "rq3_logo",
    "rq3_models",
    "tuning_selections",
]

_ENVELOPE = (3e-6, 3e-2)
_SLOPES = (
    r"$\beta_{\log\rho}$",
    r"$\beta_{\mathrm{ind}}$",
    r"$\beta_{A}$",
    r"$\beta_{\mathrm{div}}$",
)


def _worst_class(data: Dataset, name: str) -> dict[str, float]:
    """The most adverse reading of one class's preflight across the three splits."""
    cells = [preflight["by_class"][name] for preflight in data.preflights]
    return {
        "kish": min(cell["kish_effective_count"] for cell in cells),
        "p2_5": min(cell["p2_5_kish_effective_count"] for cell in cells),
        "weight": max(cell["max_patient_weight_fraction"] for cell in cells),
        "dominant": max(cell["frac_replicates_dominant"] for cell in cells),
        "flagged": any(cell["is_descriptive_only"] for cell in cells),
    }


def preflight_outcome(datasets: list[Dataset]) -> str:
    """Label-only bootstrap preflight per dataset and class."""
    rows = []
    for data in datasets:
        for name in sorted(data.preflights[0]["by_class"]):
            worst = _worst_class(data, name)
            rows.append(
                {
                    "Dataset": data.name,
                    "Class": name.replace("_", " "),
                    "Kish $n_{\\mathrm{eff}}$": num(worst["kish"], 1),
                    "2.5th pct.": num(worst["p2_5"], 1),
                    "Max unit weight": num(worst["weight"]),
                    "Dominant-unit share": num(worst["dominant"]),
                    "Descriptive-only": "Yes" if worst["flagged"] else "No",
                }
            )
    return body(pd.DataFrame(rows), longtable=True)


def _severity_spread(data: Dataset, unit: tuple[str, str]) -> float:
    """Relative spread of achieved severity across the three locked partitions."""
    assignment, severity = unit
    values = [
        freeze["assignment_conditions"][assignment][severity]["achieved_rho"]
        for freeze in data.freezes
    ]
    return (max(values) - min(values)) / min(values)


def completeness(datasets: list[Dataset]) -> str:
    """Outcome of each guard the crossed aggregation enforces before it writes."""
    rows = []
    for data in datasets:
        keys = {
            (row["assignment"], row["severity"], row["method"], row["gate"])
            for row in data.comparisons
        }
        rows.append(
            {
                "Dataset": data.name,
                "Splits": min(row["n_splits"] for row in data.comparisons),
                "Seeds per block": EXPECTED_CONFIRMATION_SEEDS,
                "Comparisons": len(keys),
                "Max severity spread": num(
                    max(_severity_spread(data, unit) for unit in comparison_units(data))
                ),
                "Descriptive-only": "No"
                if not any(row.get("descriptive_only") for row in data.comparisons)
                else "Yes",
                "Guards": "All passed",
            }
        )
    return body(pd.DataFrame(rows))


def _selection_rows(data: Dataset) -> list[dict[str, Any]]:
    rows = []
    for condition, record in sorted(data.selections[0].items()):
        for assignment, conditions in sorted(record.items()):
            for methods in conditions.values():
                rows += [
                    _selection_row(data, condition, assignment, method, values)
                    for method, values in sorted(methods.items())
                ]
    return rows


def _selection_row(
    data: Dataset,
    condition: str,
    assignment: str,
    method: str,
    values: dict[str, Any],
) -> dict[str, Any]:
    learning_rate = values.get("lr")
    budgets = data.freezes[0]["exposure_budgets"]
    return {
        "Dataset": data.name,
        "Condition": CONDITION[condition],
        "Assignment": ASSIGNMENT[assignment]
        if condition not in {"natural", "balanced"}
        else "---",
        "Method": METHOD.get(method, method),
        "Learning rate": "---" if learning_rate is None else f"{learning_rate:g}",
        "Control": num(values.get("parameter"), 4)
        if values.get("parameter") is not None
        else "---",
        "At envelope bound": "Yes" if learning_rate in _ENVELOPE else "No",
        "Example presentations": budgets[
            "natural" if condition == "natural" else "controlled"
        ],
    }


def tuning_selections(datasets: list[Dataset]) -> str:
    """Selected controls and frozen exposure budgets, with boundary flags.

    One configuration is chosen against the equal-weight three-split objective
    and written identically to every split, so split 0 carries the selection.
    """
    rows = [row for data in datasets for row in _selection_rows(data)]
    return body(pd.DataFrame(rows), longtable=True)


def per_split(datasets: list[Dataset]) -> str:
    """CE deficits partition by partition, beside the equal-weight average."""
    rows = []
    for data in datasets:
        for unit in comparison_units(data):
            for gate in ("discrimination", "calibration"):
                row = ce_row(data, unit, gate)
                if row is None:
                    continue
                effects = row["split_effects"]
                rows.append(
                    {
                        "Dataset": data.name,
                        "Assignment": ASSIGNMENT[unit[0]],
                        "Severity": CONDITION[unit[1]],
                        "Gate": gate.capitalize(),
                        "Split 0": num(effects["0"], 4),
                        "Split 1": num(effects["1"], 4),
                        "Split 2": num(effects["2"], 4),
                        "Equal weight": num(row["effect"], 4),
                        r"$\rho$ spread": num(_severity_spread(data, unit)),
                    }
                )
    return body(pd.DataFrame(rows), longtable=True)


def _diagnostics_row(
    dataset_name: str, split_index: int, row: dict[str, Any]
) -> dict[str, Any]:
    extra = {
        key.replace("_", " ").capitalize(): value
        for key, value in row.items()
        if key not in {"condition", "method", "seeds"}
    }
    return {
        "Dataset": dataset_name,
        "Split": split_index,
        "Condition": CONDITION.get(row["condition"], row["condition"]),
        "Method": METHOD.get(row["method"], row["method"]),
        "Seeds": row["seeds"],
        **extra,
    }


def method_diagnostics(datasets: list[Dataset]) -> str:
    """Per-condition, per-method rollup of any ``method_diagnostics`` counter a run recorded."""
    rows = [
        _diagnostics_row(data.name, split_index, row)
        for data in datasets
        for split_index, payload in enumerate(data.method_diagnostics)
        for row in payload.get("rows", [])
    ]
    return body(pd.DataFrame(rows), longtable=True)


def rq3_models(rq3: dict[str, Any]) -> str:
    """Standardized coefficients of the two exploratory association models."""
    rows = []
    for name, fit in rq3["models"].items():
        if not fit:
            continue
        rows.append(
            {
                "Model": name.capitalize(),
                r"$\alpha$": num(fit["intercept"], 4),
                **{
                    label: num(value, 4)
                    for label, value in zip(_SLOPES, fit["slopes"], strict=True)
                },
                r"$\sigma_u$": num(fit["sigma_u"], 4),
                r"$\sigma$": num(fit["sigma"], 4),
            }
        )
    return body(pd.DataFrame(rows))


def _group_label(group: str) -> str:
    """Report label for a dataset-target random-intercept group."""
    dataset, target = group.split(":")
    return f"{dataset.upper().replace('_', '-')} ({target.replace('_', ' ')})"


def rq3_logo(rq3: dict[str, Any]) -> str:
    """Leave-one-dataset-target-group-out held-out damage error."""
    rows = [
        {
            "Held-out group": _group_label(group),
            "Cells": entry["n"],
            "Held-out RMSE": num(entry["held_out_rmse"], 4),
        }
        for group, entry in sorted(rq3["leave_one_group_out"].items())
    ]
    return body(pd.DataFrame(rows))
