from __future__ import annotations

from pathlib import Path
from typing import cast

import pandas as pd

from imbalance_benchmark.analysis.db import connect_db
from imbalance_benchmark.analysis.query import load_eval_details
from imbalance_benchmark.analysis.reporting.equal_split import (
    classwise_table,
    tier_table,
)
from imbalance_benchmark.common import split_paths


def write_equal_split_endpoint_table(base_paths: dict[str, Path]) -> None:
    """Write canonical endpoint summaries with equal, rather than run-count, split weight."""
    equal = _equal_split_endpoints(base_paths)
    base_paths["tables"].mkdir(parents=True, exist_ok=True)
    (base_paths["tables"] / "equal_split_endpoints.tex").write_text(
        _latex_endpoint_table(equal), encoding="utf-8"
    )
    (base_paths["tables"] / "equal_split_classwise_endpoints.tex").write_text(
        classwise_table(base_paths), encoding="utf-8"
    )
    (base_paths["tables"] / "equal_split_tier_endpoints.tex").write_text(
        tier_table(base_paths), encoding="utf-8"
    )


def _equal_split_endpoints(base_paths: dict[str, Path]) -> pd.DataFrame:
    """Load test results and average first within, then equally across, splits."""
    frames = []
    for index in range(3):
        paths = split_paths(base_paths, index)
        if not paths["db"].exists():
            raise RuntimeError(
                "Every patient split must be analysed before aggregation"
            )
        conn = connect_db(paths["db"])
        try:
            frame = load_eval_details(conn)
        finally:
            conn.close()
        frame["patient_split"] = index
        frames.append(frame[frame["split"] == "test"])
    details = pd.concat(frames, ignore_index=True)
    metrics = [
        "balanced_accuracy",
        "macro_f1",
        "negative_log_likelihood",
        "macro_nll",
        "brier_score",
        "expected_calibration_error",
        "quadratic_weighted_kappa",
        "ordinal_mean_absolute_error",
        "slide_macro_balanced_accuracy",
        "patient_macro_balanced_accuracy",
        "slide_macro_f1",
        "patient_macro_f1",
        "slide_macro_nll",
        "patient_macro_nll",
        "slide_macro_brier",
        "patient_macro_brier",
        *[column for column in details if column.startswith("tier_")],
    ]
    metrics = [metric for metric in metrics if metric in details.columns]
    per_split = details.groupby(
        ["patient_split", "assignment", "condition", "method"], as_index=False
    )[metrics].mean()
    return cast(
        pd.DataFrame,
        per_split.groupby(["assignment", "condition", "method"], as_index=False)
        .mean(numeric_only=True)
        .drop(columns="patient_split"),
    )


def _latex_endpoint_table(equal: pd.DataFrame) -> str:
    """Render the canonical equal-split endpoint table as LaTeX."""
    return (
        "\\begin{table}[ht]\n\\centering\n"
        + equal.to_latex(index=False, float_format="%.3f")
        + "\\caption{Equal-weight three-split test endpoints.}\n\\label{tab:equal-split-results}\n\\end{table}\n"
    )
