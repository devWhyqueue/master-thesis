from __future__ import annotations

from typing import Any, cast

import pandas as pd

from imbalance_benchmark.analysis.inference.gates import (
    CALIBRATION_THRESHOLD,
    DISCRIMINATION_THRESHOLD,
    ci_excludes_zero,
)
from imbalance_benchmark.analysis.reporting.report.sources import (
    ASSIGNMENT,
    CONDITION,
    Dataset,
    SHORTAGE,
    body,
    ci,
    num,
    unit_key,
)

__all__ = [
    "calibration_deficit",
    "ce_row",
    "comparison_units",
    "discrimination_deficit",
    "gate_routing",
    "natural_anchor",
    "signal_profiles",
    "tier_endpoints",
]

_SEVERITY_ORDER = {"moderate": 0, "severe": 1}
_ASSIGNMENT_ORDER = {"native": 0, "difficulty_aligned": 1, "difficulty_reversed": 2}


def comparison_units(data: Dataset) -> list[tuple[str, str]]:
    """Every (tail assignment, severity) comparison unit this dataset realized."""
    units = {
        (row["assignment"], row["severity"])
        for row in data.comparisons
        if row["method"] == "ce"
    }
    return sorted(units, key=lambda u: (_SEVERITY_ORDER[u[1]], _ASSIGNMENT_ORDER[u[0]]))


def ce_row(data: Dataset, unit: tuple[str, str], gate: str) -> dict[str, Any] | None:
    """The CE gate row for one comparison unit, or None where the axis is absent."""
    return next(
        (
            row
            for row in data.comparisons
            if row["method"] == "ce"
            and row["gate"] == gate
            and (row["assignment"], row["severity"]) == unit
        ),
        None,
    )


def _frozen_mean(data: Dataset, unit: tuple[str, str], field: str) -> float:
    assignment, severity = unit
    values = [
        freeze["assignment_conditions"][assignment][severity][field]
        for freeze in data.freezes
    ]
    return sum(values) / len(values)


def _profile_mean(data: Dataset, unit: tuple[str, str], field: str) -> float:
    values = [
        entry[field]
        for profile in data.profiles
        for entry in profile["comparisons"]
        if (entry["assignment"], entry["severity"]) == unit
    ]
    return sum(values) / len(values)


def signal_profiles(datasets: list[Dataset]) -> str:
    """Realized signal profile and dominant shortage per comparison unit."""
    rows = []
    for data in datasets:
        for unit in comparison_units(data):
            record = data.units[unit_key(data.group, *unit)]
            scores = record["standardized_scores"]
            rows.append(
                {
                    "Dataset": data.name,
                    "Assignment": ASSIGNMENT[unit[0]],
                    "Severity": CONDITION[unit[1]],
                    r"$\rho$": num(_frozen_mean(data, unit, "achieved_rho"), 2),
                    r"$1-H_{\mathrm{norm}}$": num(
                        _frozen_mean(data, unit, "normalized_entropy")
                    ),
                    r"$S_{\mathrm{nom}}$": num(scores["nominal"], 2),
                    r"$S_{\mathrm{ind}}$": num(scores["independent"], 2),
                    r"$S_{\mathrm{div}}$": num(scores["diversity"], 2),
                    "$A$": num(
                        _profile_mean(data, unit, "support_difficulty_alignment"), 2
                    ),
                    "Dominant": SHORTAGE[record["dominant"]],
                }
            )
    return body(pd.DataFrame(rows))


def _gate_reason(row: dict[str, Any] | None, threshold: float) -> str:
    """Why a gate did not open, distinguishing the three closing conditions."""
    if row is None:
        return "Axis not estimable"
    if row.get("descriptive_only"):
        return "Descriptive-only"
    if row["effect"] < threshold:
        return "Below threshold"
    if not ci_excludes_zero(*row["ci"]):
        return "Interval covers zero"
    return "Open"


def _deficit_rows(
    datasets: list[Dataset], gate: str, threshold: float, digits: int
) -> list[dict[str, str]]:
    rows = []
    for data in datasets:
        for unit in comparison_units(data):
            row = ce_row(data, unit, gate)
            rows.append(
                {
                    "Dataset": data.name,
                    "Assignment": ASSIGNMENT[unit[0]],
                    "Severity": CONDITION[unit[1]],
                    "Deficit": num(row["effect"] if row else None, digits),
                    r"95\% CI": ci(row["ci"] if row else None, digits),
                    "Gate": _gate_reason(row, threshold),
                }
            )
    return rows


def discrimination_deficit(datasets: list[Dataset]) -> str:
    """Case-macro balanced-accuracy deficit and discrimination-gate outcome."""
    rows = _deficit_rows(datasets, "discrimination", DISCRIMINATION_THRESHOLD, 4)
    frame = pd.DataFrame(rows).rename(columns={"Deficit": r"$D_{\mathrm{BA}}$"})
    return body(frame)


def _ece(data: Dataset, unit: tuple[str, str]) -> tuple[str, str]:
    """Crossed patient-bootstrap ECE of this unit's imbalanced CE model."""
    table = data.tables.get("calibration_table")
    if table is None or table.empty:
        return "---", "---"
    match = table[
        (table["assignment"] == unit[0])
        & (table["condition"] == unit[1])
        & (table["method"] == "ce")
    ]
    if match.empty:
        return "---", "---"
    return num(match.iloc[0]["ECE"]), str(match.iloc[0]["ECE 95% CI"])


def calibration_deficit(datasets: list[Dataset]) -> str:
    """Tail-group macro-NLL deficit, its gate, and secondary ECE descriptors."""
    rows = _deficit_rows(datasets, "calibration", CALIBRATION_THRESHOLD, 3)
    units = [(data, unit) for data in datasets for unit in comparison_units(data)]
    for row, (data, unit) in zip(rows, units, strict=True):
        row["ECE"], row[r"ECE 95\% CI"] = _ece(data, unit)
    frame = pd.DataFrame(rows).rename(columns={"Deficit": r"$D_{\mathrm{cal}}$ (nats)"})
    return body(frame)


def _routing_row(data: Dataset, unit: tuple[str, str]) -> dict[str, str]:
    """One unit's routing state, with the reason a closed gate did not open."""
    gates = {
        gate: ce_row(data, unit, gate) for gate in ("discrimination", "calibration")
    }
    opened = [name for name, row in gates.items() if row and row["gate_passed"]]
    return {
        "Dataset": data.name,
        "Assignment": ASSIGNMENT[unit[0]],
        "Severity": CONDITION[unit[1]],
        "Discrimination": _gate_reason(
            gates["discrimination"], DISCRIMINATION_THRESHOLD
        ),
        "Probability quality": _gate_reason(
            gates["calibration"], CALIBRATION_THRESHOLD
        ),
        "Routed to": ", ".join(opened).capitalize() if opened else "Neither",
    }


def gate_routing(datasets: list[Dataset]) -> str:
    """Which of the two gates opened for each unit, and why a closed one did not."""
    rows = [
        _routing_row(data, unit) for data in datasets for unit in comparison_units(data)
    ]
    return body(pd.DataFrame(rows))


def _ce_endpoints(datasets: list[Dataset], table: str) -> pd.DataFrame:
    """Stack one endpoint table across datasets, keeping the CE reference only."""
    frames = []
    for data in datasets:
        frame = data.tables[table]
        frame = frame[frame["method"] == "ce"].copy()
        frame.insert(0, "Dataset", data.name)
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    combined["assignment"] = combined["assignment"].map(ASSIGNMENT.get)
    combined["condition"] = combined["condition"].map(CONDITION.get)
    return combined.drop(columns=["method"])


def tier_endpoints(datasets: list[Dataset]) -> str:
    """Head, body, and tail CE endpoints under every controlled condition.

    Tier membership follows the allocation of an imbalanced condition, so the
    balanced reference and the natural anchor carry no tier and are omitted.
    """
    frame = _ce_endpoints(datasets, "equal_split_tier_endpoints")
    frame = cast(pd.DataFrame, frame.dropna(subset=["recall"]))
    frame.columns = pd.Index(
        ["Dataset", "Assignment", "Condition", "Tier", "Recall", "NLL", "Brier"]
    )
    return body(frame, longtable=True)


def natural_anchor(datasets: list[Dataset]) -> str:
    """Descriptive placement of the natural anchor beside the controlled conditions."""
    frame = _ce_endpoints(datasets, "equal_split_endpoints")
    columns = {
        "Dataset": "Dataset",
        "assignment": "Assignment",
        "condition": "Condition",
        "balanced_accuracy": "BA",
        "macro_f1": r"Macro $F_1$",
        "negative_log_likelihood": "NLL",
        "macro_nll": "Macro NLL",
        "expected_calibration_error": "ECE",
    }
    selected = cast(pd.DataFrame, frame[list(columns)])
    return body(selected.rename(columns=columns))
