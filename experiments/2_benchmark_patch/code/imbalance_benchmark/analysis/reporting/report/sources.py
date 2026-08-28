from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Sequence
from typing import Any, cast

import pandas as pd

__all__ = [
    "ASSIGNMENT",
    "CONDITION",
    "Dataset",
    "METHOD",
    "SHORTAGE",
    "SIGNAL",
    "body",
    "ci",
    "dataset_roots",
    "load_dataset",
    "num",
    "pval",
    "read_latex_table",
    "unit_key",
]

ASSIGNMENT = {
    "native": "Native",
    "difficulty_aligned": "Aligned",
    "difficulty_reversed": "Reversed",
    "unassigned": "---",
}
CONDITION = {
    "natural": "Natural",
    "balanced": "Balanced",
    "moderate": "Moderate",
    "severe": "Severe",
}
SIGNAL = {
    "weighted_ce": "Prevalence",
    "class_balanced_ce": "Nominal support",
    "independent_support_ce": "Independent support",
    "pilot_difficulty_ce": "Difficulty",
    "semantic_scale_ce": "Diversity",
}
METHOD = {
    "ce": "CE",
    "balanced_sampling": "Balanced sampling",
    "logit_adjustment": "Logit adjustment (train)",
    "post_hoc_logit_adjustment": "Logit adjustment (post-hoc)",
    "crt": "cRT",
    "focal": "Focal loss",
    "cfal": "CFAL",
    "ce_soft_f1": r"CE + soft $F_1$",
    "ce_soft_mcc": "CE + soft MCC",
    "oko": "OKO",
    "independent_support_ce_matched_beta": r"Independent support (matched $\beta$)",
    "matched_vs_unmatched": "Matched vs.\\ unmatched",
    **SIGNAL,
}
SHORTAGE = {
    "nominal": "Nominal support",
    "independent": "Independent support",
    "difficulty": "Difficulty",
    "diversity": "Diversity",
    None: "Ambiguous",
}

_EMPTY = (
    "\\begin{tabular}{@{}l@{}}\n\\toprule\nNo rows.\\\\\n\\bottomrule\n\\end{tabular}\n"
)


def num(value: float | None, digits: int = 3) -> str:
    """Fixed-point cell, or an em dash where the quantity is undefined."""
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "---"
    return f"{float(value):.{digits}f}"


def ci(bounds: Sequence[float] | None, digits: int = 3) -> str:
    """Bracketed interval cell from a two-element bound pair."""
    if not bounds:
        return "---"
    return f"[{num(bounds[0], digits)}, {num(bounds[1], digits)}]"


def pval(value: float | None) -> str:
    """A p-value cell; permutation resolution bottoms out below 0.0001."""
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "---"
    return "$<$0.0001" if value < 1e-4 else f"{value:.4f}"


def unit_key(group: str, assignment: str, severity: str) -> str:
    """Matching-record key for one comparison unit."""
    return f"{group}::{assignment}::{severity}"


def body(frame: pd.DataFrame, longtable: bool = False) -> str:
    """Bare tabular body.

    The report owns the float, caption, and label because a single float spans
    all four datasets; emitting them here would duplicate every label.
    """
    if frame.empty:
        return _EMPTY
    return frame.to_latex(
        index=False,
        escape=False,
        longtable=longtable,
        na_rep="---",
        float_format="%.3f",
    )


def _cells(line: str) -> list[str]:
    return [
        cell.strip().replace("\\_", "_").replace("\\%", "%").replace("\\&", "&")
        for cell in line.strip().removesuffix("\\\\").split("&")
    ]


def _numeric(frame: pd.DataFrame) -> pd.DataFrame:
    """Restore numeric dtypes lost when a frame was rendered to LaTeX."""
    for column in frame.columns:
        converted = cast(pd.Series, pd.to_numeric(frame[column], errors="coerce"))
        if bool(converted.notna().all()):
            frame[column] = converted
    return frame


def read_latex_table(path: Path) -> pd.DataFrame:
    """Parse a machine-generated ``to_latex`` table back into a frame.

    The analysis pipeline writes these endpoint tables and keeps no CSV beside
    them, so re-reading the rendered table is the only way to reuse the numbers
    without repeating the crossed bootstrap they came from.
    """
    skip = ("\\caption", "%", "\\multicolumn")
    rows = [
        _cells(line)
        for line in path.read_text().splitlines()
        if "&" in line and not line.lstrip().startswith(skip)
    ]
    header = rows[0]
    width = len(header)
    return _numeric(
        pd.DataFrame(
            [row for row in rows[1:] if len(row) == width and row != header],
            columns=pd.Index(header),
        )
    )


def _split_json(root: Path, name: str) -> list[dict[str, Any]]:
    return [
        json.loads((root / f"split={index}" / "data" / name).read_text())
        for index in range(3)
    ]


@dataclass(frozen=True)
class Dataset:
    """Every frozen artifact one dataset root contributes to the report."""

    name: str
    root: Path
    comparisons: list[dict[str, Any]]
    units: dict[str, dict[str, Any]]
    freezes: list[dict[str, Any]]
    preflights: list[dict[str, Any]]
    profiles: list[dict[str, Any]]
    selections: list[dict[str, dict[str, Any]]]
    tables: dict[str, pd.DataFrame]

    @property
    def group(self) -> str:
        """Dataset-target group key used by the matching record and RQ3."""
        provenance = self.freezes[0].get("dataset_provenance", {})
        return f"{provenance.get('name')}:{provenance.get('target')}"


def _scalar_comparisons(root: Path) -> list[dict[str, Any]]:
    """Crossed comparisons without their stored bootstrap vectors.

    Those vectors are 10,000 floats per row and dominate the file; the report
    needs only the point estimates, intervals, and test outcomes.
    """
    path = root / "data" / "cross_split_gates_and_recovery.json"
    payload = json.loads(path.read_text())
    return [
        {key: value for key, value in row.items() if not key.startswith("bootstrap")}
        for row in payload["comparisons"]
    ]


def _selections(root: Path) -> list[dict[str, dict[str, Any]]]:
    conditions = ("natural", "balanced", "moderate", "severe")
    records = []
    for index in range(3):
        data = root / f"split={index}" / "data"
        records.append(
            {
                condition: json.loads(path.read_text())
                for condition in conditions
                if (path := data / f"tuning_selections_{condition}.json").exists()
            }
        )
    return records


def load_dataset(root: Path) -> Dataset:
    """Load one dataset root's frozen analysis artifacts."""
    tables = {
        path.stem: read_latex_table(path)
        for path in sorted((root / "tables").glob("*.tex"))
    }
    record = json.loads((root / "data" / "matching_record.json").read_text())
    return Dataset(
        name=root.parent.name.upper().replace("_", "-"),
        root=root,
        comparisons=_scalar_comparisons(root),
        units=record["units"],
        freezes=_split_json(root, "manifest_freeze.json"),
        preflights=_split_json(root, "bootstrap_preflight.json"),
        profiles=_split_json(root, "signal_profile.json"),
        selections=_selections(root),
        tables=tables,
    )


def dataset_roots(config: dict[str, Any]) -> list[Path]:
    """The four dataset roots the report combines into one table per float."""
    roots = config.get("rq3", {}).get("dataset_roots", [])
    if not roots:
        raise RuntimeError(
            "report-tables needs rq3.dataset_roots listing every dataset"
        )
    return [Path(root) for root in roots]
