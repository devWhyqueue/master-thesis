from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from scripts.common import ensure_dirs, load_config
from scripts.analysis.results import (
    connect,
    init_schema,
    load_split_payload,
    load_summary,
    replace_table,
    write_json_table,
)
from scripts.analysis.report.calibration.utils import (
    calibrated_probabilities,
    fit_temperature,
    metric_bundle,
    probabilities_to_logits,
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
    """Parse post-hoc calibration table arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    return parser.parse_args()


def _settings(config: dict[str, Any], benchmark: str) -> tuple[list[str], list[int]]:
    if benchmark == "patch":
        return (
            list(config["patch_feature_methods"]),
            list(config["patch_feature_training"]["seeds"]),
        )
    return (
        list(config["wsi_bag_methods"]),
        list(config["wsi_training"]["seeds"]),
    )


def _arrays(payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, int]:
    labels = np.asarray(payload["labels"], dtype=np.int64)
    probabilities = np.asarray(payload["probabilities"], dtype=np.float64)
    n_classes = len(payload["class_names"])
    return labels, probabilities, n_classes


def _collect(
    config: dict[str, Any], paths: dict[str, Path]
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    connection = connect(paths["db"])
    init_schema(connection)
    for benchmark in ("patch", "wsi_bag"):
        methods, seeds = _settings(config, benchmark)
        for method in methods:
            for seed in seeds:
                val_payload = load_split_payload(
                    connection, benchmark, method, seed, "val"
                )
                test_payload = load_split_payload(
                    connection, benchmark, method, seed, "test"
                )
                if val_payload is None or test_payload is None:
                    missing.append(
                        {"benchmark": benchmark, "method": method, "seed": seed}
                    )
                    continue
                if "labels" not in val_payload or "labels" not in test_payload:
                    missing.append(
                        {"benchmark": benchmark, "method": method, "seed": seed}
                    )
                    continue
                val_labels, val_probs, _ = _arrays(val_payload)
                test_labels, test_probs, n_classes = _arrays(test_payload)
                fit = fit_temperature(probabilities_to_logits(val_probs), val_labels)
                calibrated_test = calibrated_probabilities(
                    probabilities_to_logits(test_probs), fit
                )
                metrics = metric_bundle(calibrated_test, test_labels, n_classes)
                rows.append(
                    {
                        "benchmark": benchmark,
                        "method": method,
                        "seed": seed,
                        "calibrator": "temperature",
                        **metrics,
                    }
                )
    connection.close()
    return pd.DataFrame(rows), missing


def _aggregate(frame: pd.DataFrame, paths: dict[str, Path]) -> pd.DataFrame:
    grouped = cast(
        pd.DataFrame,
        frame.groupby(["benchmark", "method"], as_index=False).agg(
            **{f"{metric}_mean": (metric, "mean") for metric in CALIBRATION_METRICS},
            **{f"{metric}_std": (metric, "std") for metric in CALIBRATION_METRICS},
        ),
    )
    connection = connect(paths["db"])
    init_schema(connection)
    ranks = []
    for benchmark in ("patch", "wsi_bag"):
        summary = load_summary(connection, benchmark, "test")
        if summary.empty:
            continue
        test = summary[["method", "macro_f1_mean"]]
        ranks.append(test.assign(benchmark=benchmark))
    connection.close()
    if not ranks:
        return grouped
    rank_frame = pd.concat(ranks, ignore_index=True)
    merged = grouped.merge(rank_frame, on=["benchmark", "method"], how="left")
    rows = sorted(
        merged.iterrows(),
        key=lambda item: (item[1]["benchmark"], -float(item[1]["macro_f1_mean"])),
    )
    return cast(
        pd.DataFrame,
        pd.DataFrame([row for _, row in rows]).drop(columns=["macro_f1_mean"]),
    )


def _format_metric(row: pd.Series, metric: str) -> str:
    mean = float(row[f"{metric}_mean"])
    std = float(row[f"{metric}_std"])
    if pd.isna(std):
        return rf"$\num{{{mean:.3f}}}$"
    return rf"$\num{{{mean:.3f}}} \pm \num{{{std:.3f}}}$"


def write_latex(frame: pd.DataFrame, path: Path) -> None:
    """Write LaTeX table for temperature-calibrated metrics."""
    lines = [
        "\\begin{tabular}{llccc}",
        "\\toprule",
        "Regime & Method & NLL & Brier & ECE\\\\",
        "\\midrule",
    ]
    current = ""
    for row in frame.to_dict("records"):
        benchmark = str(row["benchmark"])
        if benchmark != current:
            if current:
                lines.append("\\addlinespace")
            current = benchmark
        lines.append(
            f"{REGIME_LABELS.get(benchmark, benchmark)} & "
            f"{latex_method_label(str(row['method']))} & "
            f"{_format_metric(pd.Series(row), 'negative_log_likelihood')} & "
            f"{_format_metric(pd.Series(row), 'brier_score')} & "
            f"{_format_metric(pd.Series(row), 'expected_calibration_error')}\\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Build temperature-calibrated test calibration table for all methods."""
    args = parse_args()
    config = load_config(args.config)
    paths = ensure_dirs(config)
    frame, missing = _collect(config, paths)
    if frame.empty:
        raise RuntimeError(
            "No calibration rows collected; missing val/test result files."
        )
    aggregate = _aggregate(frame, paths)
    connection = connect(paths["db"])
    init_schema(connection)
    replace_table(connection, "calibration_posthoc", aggregate)
    write_json_table(connection, "calibration_posthoc_missing", {"missing": missing})
    connection.close()
    write_latex(aggregate, paths["tables"] / "result_calibration_posthoc.tex")


if __name__ == "__main__":
    main()
