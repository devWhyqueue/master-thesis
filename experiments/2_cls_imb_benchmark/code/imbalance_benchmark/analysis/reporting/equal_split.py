from __future__ import annotations

from pathlib import Path
from typing import cast

import pandas as pd

from imbalance_benchmark.analysis.db import connect_db
from imbalance_benchmark.analysis.query import load_classwise, load_eval_details
from imbalance_benchmark.common import split_paths


def classwise_table(base_paths: dict[str, Path]) -> str:
    """Render per-class test endpoints with equal weight for each patient split."""
    frames = _split_frames(base_paths, load_classwise)
    details = pd.concat(frames, ignore_index=True)
    metrics = ["recall", "f1", "nll", "brier"]
    keys = ["assignment", "condition", "method", "class_name", "tier"]
    per_split = details.groupby([*keys, "patient_split"], as_index=False)[
        metrics
    ].mean()
    equal = per_split.groupby(keys, as_index=False)[metrics].mean()
    return _latex(
        cast(pd.DataFrame, equal),
        "Equal-weight three-split per-class and tier test endpoints.",
        "tab:equal-split-classwise",
    )


def tier_table(base_paths: dict[str, Path]) -> str:
    """Render direct head/body/tail recall, NLL, and Brier endpoints."""
    details = pd.concat(_split_frames(base_paths, load_eval_details), ignore_index=True)
    rows = [_tier_rows(details, tier) for tier in ("head", "body", "tail")]
    rows = [row for row in rows if row is not None]
    columns = ["assignment", "condition", "method", "tier", "recall", "nll", "brier"]
    if not rows:
        equal = pd.DataFrame(columns=pd.Index(columns))
    else:
        stacked = pd.concat(rows, ignore_index=True)
        keys = ["patient_split", "assignment", "condition", "method", "tier"]
        per_split = stacked.groupby(keys, as_index=False)[
            ["recall", "nll", "brier"]
        ].mean()
        equal = per_split.groupby(keys[1:], as_index=False)[
            ["recall", "nll", "brier"]
        ].mean()
    return _latex(
        cast(pd.DataFrame, equal),
        "Equal-weight three-split head, body, and tail recall, NLL, and Brier endpoints.",
        "tab:equal-split-tier-endpoints",
    )


def _split_frames(base_paths: dict[str, Path], loader: object) -> list[pd.DataFrame]:
    frames = []
    for index in range(3):
        paths = split_paths(base_paths, index)
        if not paths["db"].exists():
            raise RuntimeError(
                "Every patient split must be analysed before aggregation"
            )
        conn = connect_db(paths["db"])
        try:
            frame = loader(conn)  # type: ignore[operator]
        finally:
            conn.close()
        frame["patient_split"] = index
        frames.append(cast(pd.DataFrame, frame[frame["split"] == "test"]))
    return frames


def _tier_rows(details: pd.DataFrame, tier: str) -> pd.DataFrame | None:
    source = [f"tier_{tier}_{metric}" for metric in ("recall", "nll", "brier")]
    if not all(column in details for column in source):
        return None
    rows = details[
        ["patient_split", "assignment", "condition", "method", *source]
    ].copy()
    rows["tier"] = tier
    rows.columns = pd.Index(
        [
            "patient_split",
            "assignment",
            "condition",
            "method",
            "recall",
            "nll",
            "brier",
            "tier",
        ]
    )
    return cast(pd.DataFrame, rows)


def _latex(frame: pd.DataFrame, caption: str, label: str) -> str:
    """Wrap one endpoint frame in the report's standard LaTeX table envelope."""
    return (
        "\\begin{table}[ht]\n\\centering\n"
        + frame.to_latex(index=False, float_format="%.3f")
        + f"\\caption{{{caption}}}\n\\label{{{label}}}\n\\end{{table}}\n"
    )
