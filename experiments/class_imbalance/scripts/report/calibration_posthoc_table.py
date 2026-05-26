from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.common import ensure_dirs, load_config
from scripts.report.calibration_utils import (
    calibrated_probabilities,
    fit_temperature,
    metric_bundle,
    probabilities_to_logits,
)
from scripts.report.figures import latex_method_label

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


def _settings(
    config: dict[str, Any], benchmark: str
) -> tuple[list[str], list[int], Path]:
    if benchmark == "patch":
        return (
            list(config["patch_feature_methods"]),
            list(config["patch_feature_training"]["seeds"]),
            Path("results") / "patch_feature",
        )
    return (
        list(config["wsi_bag_methods"]),
        list(config["wsi_training"]["seeds"]),
        Path("results_wsi_bag"),
    )


def _load_split(result_dir: Path, split: str) -> dict[str, Any] | None:
    path = result_dir / f"{split}_results.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


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
    for benchmark in ("patch", "wsi_bag"):
        methods, seeds, relative_root = _settings(config, benchmark)
        root = paths["root"] / "outputs" / relative_root
        for method in methods:
            for seed in seeds:
                result_dir = root / method / f"seed={seed}"
                val_payload = _load_split(result_dir, "val")
                test_payload = _load_split(result_dir, "test")
                if val_payload is None or test_payload is None:
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
    return pd.DataFrame(rows), missing


def _aggregate(frame: pd.DataFrame, paths: dict[str, Path]) -> pd.DataFrame:
    grouped = frame.groupby(["benchmark", "method"], as_index=False).agg(
        **{f"{metric}_mean": (metric, "mean") for metric in CALIBRATION_METRICS},
        **{f"{metric}_std": (metric, "std") for metric in CALIBRATION_METRICS},
    )
    ranks = []
    for benchmark in ("patch", "wsi_bag"):
        summary = pd.read_csv(paths["tables"] / f"result_summary_{benchmark}.csv")
        test = summary[summary["split"] == "test"][["method", "macro_f1_mean"]]
        ranks.append(test.assign(benchmark=benchmark))
    rank_frame = pd.concat(ranks, ignore_index=True)
    merged = grouped.merge(rank_frame, on=["benchmark", "method"], how="left")
    rows = sorted(
        merged.iterrows(),
        key=lambda item: (item[1]["benchmark"], -float(item[1]["macro_f1_mean"])),
    )
    return pd.DataFrame([row for _, row in rows]).drop(columns=["macro_f1_mean"])


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
    aggregate.to_csv(paths["tables"] / "result_calibration_posthoc.csv", index=False)
    write_latex(aggregate, paths["tables"] / "result_calibration_posthoc.tex")
    (paths["tables"] / "result_calibration_posthoc_missing.json").write_text(
        json.dumps({"missing": missing}, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
