from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast

import pandas as pd

from scripts.common import ensure_dirs, load_config
from scripts.analysis.results import (
    connect,
    init_schema,
    load_eval_details,
    load_summary,
    replace_table,
)
from scripts.analysis.report.figures.labels import latex_method_label

CALIBRATION_METRICS = (
    "negative_log_likelihood",
    "brier_score",
    "expected_calibration_error",
)
REGIME_LABELS = {
    "patch": "Patch",
    "wsi_bag": "WSI bag",
}


def parse_args() -> argparse.Namespace:
    """Parse calibration-table generation arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    return parser.parse_args()


def _collect_details(paths: dict[str, Path], config: dict[str, Any]) -> pd.DataFrame:
    connection = connect(paths["db"])
    init_schema(connection)
    rows: list[dict[str, Any]] = []
    for benchmark, methods_key, seeds_key in (
        ("patch", "patch_feature_methods", "patch_feature_training"),
        ("wsi_bag", "wsi_bag_methods", "wsi_training"),
    ):
        methods = list(config[methods_key])
        seeds = list(config[seeds_key]["seeds"])
        for payload in load_eval_details(connection, benchmark, methods, seeds, "test"):
            result = payload["result"]
            rows.append(
                {
                    "benchmark": benchmark,
                    "method": payload["method"],
                    **{metric: result[metric] for metric in CALIBRATION_METRICS},
                }
            )
    connection.close()
    return pd.DataFrame(rows)


def _aggregate_details(frame: pd.DataFrame, paths: dict[str, Path]) -> pd.DataFrame:
    grouped = frame.groupby(["benchmark", "method"], as_index=False).agg(
        **{f"{metric}_mean": (metric, "mean") for metric in CALIBRATION_METRICS},
        **{f"{metric}_std": (metric, "std") for metric in CALIBRATION_METRICS},
    )
    connection = connect(paths["db"])
    init_schema(connection)
    rank_frames = []
    for benchmark in ("patch", "wsi_bag"):
        summary = load_summary(connection, benchmark, "test")
        if summary.empty:
            continue
        test = cast(
            pd.DataFrame,
            summary[["method", "macro_f1_mean"]],
        )
        rank_frames.append(test.assign(benchmark=benchmark))
    connection.close()
    if not rank_frames:
        return grouped
    ranks = pd.concat(rank_frames, ignore_index=True)
    merged = grouped.merge(ranks, on=["benchmark", "method"], how="left")
    rows = sorted(
        merged.iterrows(),
        key=lambda item: (item[1]["benchmark"], -float(item[1]["macro_f1_mean"])),
    )
    return pd.DataFrame([row for _, row in rows]).drop(columns=["macro_f1_mean"])


def _method_label(method: str) -> str:
    return latex_method_label(method)


def _format_metric(row: pd.Series, metric: str) -> str:
    mean = float(row[f"{metric}_mean"])
    std = float(row[f"{metric}_std"])
    if pd.isna(std):
        return rf"$\num{{{mean:.3f}}}$"
    return rf"$\num{{{mean:.3f}}} \pm \num{{{std:.3f}}}$"


def write_calibration_latex(frame: pd.DataFrame, path: Path) -> None:
    """Write a combined patch and WSI calibration table."""
    lines = [
        "\\begin{tabular}{llccc}",
        "\\toprule",
        "Regime & Method & NLL & Brier & ECE\\\\",
        "\\midrule",
    ]
    current_benchmark = ""
    for row in frame.to_dict("records"):
        benchmark = str(row["benchmark"])
        if benchmark != current_benchmark:
            if current_benchmark:
                lines.append("\\addlinespace")
            current_benchmark = benchmark
        regime = REGIME_LABELS.get(benchmark, benchmark)
        lines.append(
            f"{regime} & {_method_label(str(row['method']))} & "
            f"{_format_metric(pd.Series(row), 'negative_log_likelihood')} & "
            f"{_format_metric(pd.Series(row), 'brier_score')} & "
            f"{_format_metric(pd.Series(row), 'expected_calibration_error')}\\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Build the combined calibration table from stored result details."""
    args = parse_args()
    config = load_config(args.config)
    paths = ensure_dirs(config)
    aggregate = _aggregate_details(_collect_details(paths, config), paths)
    connection = connect(paths["db"])
    init_schema(connection)
    replace_table(connection, "calibration", aggregate)
    connection.close()
    write_calibration_latex(aggregate, paths["tables"] / "result_calibration.tex")


if __name__ == "__main__":
    main()
