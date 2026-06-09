from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any, cast

import pandas as pd

from scripts.common import ensure_dirs, load_config
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


def _read_details(path: Path, benchmark: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            if payload["split"] != "test":
                continue
            result = payload["result"]
            rows.append(
                {
                    "benchmark": benchmark,
                    "method": payload["method"],
                    **{metric: result[metric] for metric in CALIBRATION_METRICS},
                }
            )
    return pd.DataFrame(rows)


def _aggregate_details(frame: pd.DataFrame, paths: dict[str, Path]) -> pd.DataFrame:
    grouped = frame.groupby(["benchmark", "method"], as_index=False).agg(
        **{f"{metric}_mean": (metric, "mean") for metric in CALIBRATION_METRICS},
        **{f"{metric}_std": (metric, "std") for metric in CALIBRATION_METRICS},
    )
    rank_frames = []
    for benchmark in ("patch", "wsi_bag"):
        summary_path = paths["tables"] / f"result_summary_{benchmark}.csv"
        summary = pd.read_csv(summary_path)
        test = cast(
            pd.DataFrame,
            summary[summary["split"] == "test"][["method", "macro_f1_mean"]],
        )
        rank_frames.append(test.assign(benchmark=benchmark))
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
    frames = [
        _read_details(paths["tables"] / "result_details_patch.jsonl.gz", "patch"),
        _read_details(paths["tables"] / "result_details_wsi_bag.jsonl.gz", "wsi_bag"),
    ]
    aggregate = _aggregate_details(pd.concat(frames, ignore_index=True), paths)
    aggregate.to_csv(paths["tables"] / "result_calibration.csv", index=False)
    write_calibration_latex(aggregate, paths["tables"] / "result_calibration.tex")


if __name__ == "__main__":
    main()
