from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd


def load_rq3_cells(analysis_roots: list[Path]) -> list[dict[str, Any]]:
    """Use crossed-bootstrap outcomes and equal split weights for RQ3 cells."""
    cells = []
    for root in analysis_roots:
        split_cells = _load_all_splits(root)
        cells.extend(_attach_crossed_outcomes(root, split_cells))
    if not cells:
        return []
    frame = pd.DataFrame(cells)
    if "gate" not in frame:
        frame["gate"] = "discrimination"
    keys = ["group", "assignment", "severity", "method", "gate"]
    numeric = [
        column
        for column in frame.select_dtypes(include=["number"])
        if column not in keys
    ]
    averaged = cast(pd.DataFrame, frame.groupby(keys, as_index=False)[numeric].mean())
    averaged["gate_passed"] = frame.groupby(keys)["gate_passed"].first().to_numpy()
    return averaged.to_dict(orient="records")


def _load_all_splits(root: Path) -> list[dict[str, Any]]:
    cells = []
    for index in range(3):
        path = root / f"split={index}" / "data" / "rq3.json"
        if not path.exists():
            raise RuntimeError(
                f"RQ3 requires all three completed patient splits; missing {path}"
            )
        cells.extend(json.loads(path.read_text()).get("cells", []))
    return cells


def _attach_crossed_outcomes(
    root: Path, cells: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    path = root / "data" / "cross_split_gates_and_recovery.json"
    if not path.exists():
        raise RuntimeError(f"RQ3 requires crossed split aggregation: missing {path}")
    comparisons = json.loads(path.read_text()).get("comparisons", [])
    gates, outcomes = _comparison_maps(comparisons)
    missing = {_key(cell) for cell in cells} - set(outcomes)
    if missing:
        raise RuntimeError(
            f"Crossed RQ3 aggregation is missing comparison cells: {sorted(missing)}"
        )
    return [_crossed_cell(cell, gates, outcomes[_key(cell)]) for cell in cells]


def _comparison_maps(
    rows: list[dict[str, Any]],
) -> tuple[
    dict[tuple[str, str, str], bool], dict[tuple[str, str, str, str], dict[str, Any]]
]:
    gates: dict[tuple[str, str, str], bool] = {}
    outcomes = {}
    for row in rows:
        assignment, severity, gate = row["assignment"], row["severity"], row["gate"]
        if row["method"] == "ce":
            gates[(assignment, severity, gate)] = gates.get(
                (assignment, severity, gate), False
            ) or bool(row.get("gate_passed"))
        outcomes[(assignment, severity, row["method"], gate)] = row
    return gates, outcomes


def _key(cell: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        cell["assignment"],
        cell["severity"],
        cell["method"],
        cell.get("gate", "discrimination"),
    )


def _crossed_cell(
    cell: dict[str, Any], gates: dict[tuple[str, str, str], bool], row: dict[str, Any]
) -> dict[str, Any]:
    updated = dict(cell)
    gate_key = cell["assignment"], cell["severity"], cell.get("gate", "discrimination")
    updated["gate_passed"] = (
        any(
            passed
            for (assignment, severity, _), passed in gates.items()
            if (assignment, severity) == gate_key[:2]
        )
        if cell["method"] == "ce"
        else gates[gate_key]
    )
    if cell["method"] == "ce":
        # Replicate 0 is the observed cross-split deficit (equal split weight);
        # replicates 1.. supply only the bootstrap spread. Match aggregate.py.
        effects = np.asarray(row["bootstrap_effect"], dtype=float)
        updated["deficit_ba"] = float(effects[0])
        updated["deficit_se"] = float(np.nanstd(effects, ddof=1))
    elif "bootstrap_numerator" in row and "bootstrap_denominator" in row:
        numerator = np.asarray(row["bootstrap_numerator"], dtype=float)
        denominator = np.asarray(row["bootstrap_denominator"], dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            recovery = np.where(denominator != 0, numerator / denominator, np.nan)
        updated["recovery"] = (
            float(numerator[0] / denominator[0]) if denominator[0] != 0 else np.nan
        )
        updated["recovery_se"] = float(np.nanstd(recovery, ddof=1))
    return updated
