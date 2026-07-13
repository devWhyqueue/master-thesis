from __future__ import annotations

import sqlite3
from typing import Any, cast

import pandas as pd

from imbalance_benchmark.analysis.query import load_classwise, load_eval_details

__all__ = ["results_table", "calibration_table", "confirmatory_table", "rq3_table"]


def _to_latex(df: pd.DataFrame, caption: str, label: str) -> str:
    """Render a small results frame as a captioned LaTeX table."""
    if df.empty:
        body = "\\multicolumn{1}{c}{No confirmed runs ingested yet.}"
    else:
        body = df.to_latex(index=False, float_format="%.3f", escape=True)
    return (
        f"% {caption}\n"
        "\\begin{table}[ht]\n\\centering\n"
        f"{body}\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n\\end{{table}}\n"
    )


def _with_tail_recall(
    summary: pd.DataFrame, conn: sqlite3.Connection, split: str
) -> pd.DataFrame:
    """Merge in each condition/method's mean tail-tier recall, when classwise rows exist."""
    classwise = load_classwise(conn)
    classwise = cast(pd.DataFrame, classwise[classwise["split"] == split])
    if classwise.empty:
        return summary
    tail_series = cast(
        pd.Series,
        classwise[classwise["tier"] == "tail"]
        .groupby(["condition", "method"])["recall"]
        .mean(),
    )
    tail = tail_series.rename("tail_recall").reset_index()
    return summary.merge(tail, on=["condition", "method"], how="left")


def results_table(conn: sqlite3.Connection, split: str = "test") -> str:
    """Real BA/macro-F1/tail-recall/NLL/macro-NLL table by condition x method (replaces the placeholder)."""
    details = load_eval_details(conn)
    details = details[details["split"] == split]
    if details.empty:
        return _to_latex(pd.DataFrame(), "Confirmation results", "tab:results")
    summary = (
        details.groupby(["condition", "method"])
        .agg(
            balanced_accuracy=("balanced_accuracy", "mean"),
            macro_f1=("macro_f1", "mean"),
            negative_log_likelihood=("negative_log_likelihood", "mean"),
            macro_nll=("macro_nll", "mean"),
            n_seeds=("seed_index", "nunique"),
        )
        .reset_index()
    )
    summary = _with_tail_recall(summary, conn, split)
    return _to_latex(
        summary, "Confirmation results by condition and method", "tab:results"
    )


def calibration_table(conn: sqlite3.Connection, split: str = "test") -> str:
    """Raw vs. temperature-scaled NLL/Brier/ECE by condition x method."""
    details = load_eval_details(conn)
    details = details[details["split"] == split]
    if details.empty:
        return _to_latex(pd.DataFrame(), "Calibration summary", "tab:calibration")
    summary = (
        details.groupby(["condition", "method"])
        .agg(
            negative_log_likelihood=("negative_log_likelihood", "mean"),
            brier_score=("brier_score", "mean"),
            expected_calibration_error=("expected_calibration_error", "mean"),
        )
        .reset_index()
    )
    return _to_latex(summary, "Raw calibration summary", "tab:calibration")


def confirmatory_table(comparisons: list[dict[str, Any]]) -> str:
    """Confirmatory-family table with Holm-adjusted p-values and "not tested" cells."""
    df = pd.DataFrame(comparisons)
    if df.empty:
        return _to_latex(df, "Confirmatory family results", "tab:confirmatory")
    cols = [
        c
        for c in [
            "method",
            "gate",
            "severity",
            "effect",
            "p_value",
            "adjusted_p_value",
            "status",
        ]
        if c in df.columns
    ]
    return _to_latex(
        cast(pd.DataFrame, df[cols]),
        "Confirmatory family results (Holm-adjusted)",
        "tab:confirmatory",
    )


def rq3_table(models: dict[str, dict[str, Any]]) -> str:
    """RQ3 coefficient table: intercept and (log rho, separability) slopes per linked model."""
    rows = []
    for name, fit in models.items():
        if not fit:
            continue
        slopes = fit.get("slopes", [None, None])
        rows.append(
            {
                "model": name,
                "intercept": fit.get("intercept"),
                "slope_log_rho": slopes[0] if len(slopes) > 0 else None,
                "slope_separability": slopes[1] if len(slopes) > 1 else None,
                "sigma_u": fit.get("sigma_u"),
                "sigma": fit.get("sigma"),
            }
        )
    return _to_latex(pd.DataFrame(rows), "RQ3 linked exploratory models", "tab:rq3")
