from __future__ import annotations

from typing import Any, cast

import pandas as pd

from imbalance_benchmark.analysis.inference.confirmatory.holm import PRIMARY_METHODS
from imbalance_benchmark.analysis.reporting.report.rq1 import ce_row, comparison_units
from imbalance_benchmark.analysis.reporting.report.sources import (
    ASSIGNMENT,
    CONDITION,
    METHOD,
    SHORTAGE,
    SIGNAL,
    Dataset,
    body,
    ci,
    num,
    pval,
    unit_key,
)

__all__ = [
    "calibration_recovery",
    "confirmatory",
    "matched_beta",
    "matched_contrast",
    "roster_recovery",
]

_SIGNAL_ORDER = list(SIGNAL)


def _rows(data: Dataset, unit: tuple[str, str], gate: str) -> dict[str, dict[str, Any]]:
    """This unit's crossed comparison rows for one gate, keyed by method."""
    return {
        row["method"]: row
        for row in data.comparisons
        if row["gate"] == gate and (row["assignment"], row["severity"]) == unit
    }


def _open_units(
    datasets: list[Dataset], gate: str
) -> list[tuple[Dataset, tuple[str, str]]]:
    """Every unit whose CE gate opened on this axis, in report order."""
    return [
        (data, unit)
        for data in datasets
        for unit in comparison_units(data)
        if (row := ce_row(data, unit, gate)) and row["gate_passed"]
    ]


def _recovery_cells(row: dict[str, Any] | None) -> dict[str, str]:
    return {
        "$R_M$": num(row.get("recovery") if row else None),
        r"95\% CI": ci(row.get("recovery_ci") if row else None),
        "$p$": pval(row.get("p_value") if row else None),
        "Holm $p$": pval(row.get("adjusted_p_value") if row else None),
        "Status": (row or {}).get("status", "---").capitalize(),
    }


def _family_frame(datasets: list[Dataset], gate: str) -> pd.DataFrame:
    rows = []
    for data, unit in _open_units(datasets, gate):
        methods = _rows(data, unit, gate)
        for method in _SIGNAL_ORDER:
            rows.append(
                {
                    "Dataset": data.name,
                    "Assignment": ASSIGNMENT[unit[0]],
                    "Severity": CONDITION[unit[1]],
                    "Signal": SIGNAL[method],
                    **_recovery_cells(methods.get(method)),
                }
            )
    return pd.DataFrame(rows)


def confirmatory(datasets: list[Dataset]) -> str:
    """Five-family discrimination recovery on discrimination-gate-passing units."""
    return body(_family_frame(datasets, "discrimination"), longtable=True)


def calibration_recovery(datasets: list[Dataset]) -> str:
    """Tail-NLL recovery on calibration-gate units, beside the same cell's BA recovery."""
    frame = _family_frame(datasets, "calibration")
    paired = []
    for data, unit in _open_units(datasets, "calibration"):
        discrimination = _rows(data, unit, "discrimination")
        opened = (ce_row(data, unit, "discrimination") or {}).get("gate_passed")
        paired += [
            num(discrimination.get(method, {}).get("recovery")) if opened else "---"
            for method in _SIGNAL_ORDER
        ]
    frame["Discrimination $R_M$"] = paired
    return body(frame, longtable=True)


def _best_signal(methods: dict[str, dict[str, Any]]) -> str:
    """Which family member in fact recovered most in this unit."""
    scored = {
        method: methods[method]["recovery"]
        for method in _SIGNAL_ORDER
        if methods.get(method) and methods[method].get("recovery") is not None
    }
    if not scored:
        return "---"
    return SIGNAL[max(scored, key=lambda method: scored[method])]


def _contrast_row(data: Dataset, unit: tuple[str, str], gate: str) -> dict[str, str]:
    """One gate-passing unit's matched-versus-unmatched contrast row."""
    methods = _rows(data, unit, gate)
    record = data.units[unit_key(data.group, *unit)]
    contrast = methods.get("matched_vs_unmatched")
    best = _best_signal(methods)
    matched = [
        SIGNAL[method] for method in record["matched_methods"] if method in SIGNAL
    ]
    return {
        "Dataset": data.name,
        "Assignment": ASSIGNMENT[unit[0]],
        "Severity": CONDITION[unit[1]],
        "Gate": gate.capitalize(),
        "Dominant": SHORTAGE[record["dominant"]],
        "Contrast": num(contrast["effect"] if contrast else None, 4),
        r"95\% CI": ci(contrast["ci"] if contrast else None, 4),
        "Holm $p$": pval(contrast.get("adjusted_p_value") if contrast else None),
        "Best recoverer": best,
        "Agrees": "---" if not matched else ("Yes" if best in matched else "No"),
    }


def matched_contrast(datasets: list[Dataset]) -> str:
    """Direct matched-versus-unmatched contrast per gate-passing unit."""
    rows = [
        _contrast_row(data, unit, gate)
        for gate in ("discrimination", "calibration")
        for data, unit in _open_units(datasets, gate)
    ]
    return body(pd.DataFrame(rows), longtable=True)


_BETA_METHODS = {
    "ce": "CE",
    "independent_support_ce": "Independent",
    "independent_support_ce_matched_beta": r"Independent at matched $\beta$",
    "class_balanced_ce": "Nominal",
}


def matched_beta(datasets: list[Dataset]) -> str:
    """Balanced accuracy of the matched-beta diagnostic arm and its two anchors."""
    rows = []
    for data in datasets:
        table = data.tables["equal_split_endpoints"]
        for unit in comparison_units(data):
            cell = cast(
                pd.DataFrame,
                table[
                    (table["assignment"] == unit[0]) & (table["condition"] == unit[1])
                ],
            )
            entry = {
                "Dataset": data.name,
                "Assignment": ASSIGNMENT[unit[0]],
                "Severity": CONDITION[unit[1]],
            }
            for method, label in _BETA_METHODS.items():
                match = cast(pd.DataFrame, cell[cell["method"] == method])
                entry[label] = (
                    num(match.iloc[0]["balanced_accuracy"])
                    if not match.empty
                    else "---"
                )
            rows.append(entry)
    return body(pd.DataFrame(rows))


def roster_recovery(datasets: list[Dataset]) -> str:
    """Exploratory recovery for roster members that also vary the mechanism."""
    rows = []
    for gate in ("discrimination", "calibration"):
        for data, unit in _open_units(datasets, gate):
            methods = _rows(data, unit, gate)
            exploratory = sorted(
                method
                for method in methods
                if method not in PRIMARY_METHODS
                and method not in {"ce", "matched_vs_unmatched"}
            )
            for method in exploratory:
                cells = _recovery_cells(methods[method])
                rows.append(
                    {
                        "Dataset": data.name,
                        "Assignment": ASSIGNMENT[unit[0]],
                        "Severity": CONDITION[unit[1]],
                        "Gate": gate.capitalize(),
                        "Method": METHOD.get(method, method),
                        "$R_M$": cells["$R_M$"],
                        r"95\% CI": cells[r"95\% CI"],
                    }
                )
    return body(pd.DataFrame(rows), longtable=True)
