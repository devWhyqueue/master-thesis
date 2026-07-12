"""Build report tables and figures for the native CAMELYON16 benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import pandas as pd

from analysis.plotting import (
    PATCH_ORDER,
    WSI_ORDER,
    _benchmark,
    _mean_std,
    _method_label,
    _write_table,
    _write_unavailable,
)
from analysis.plotting.calibration import _calibration_row

LABELS = ("normal", "tumor")


def parse_args() -> argparse.Namespace:
    """Parse CAMELYON16 report aggregation arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-report", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--selection-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    """Write native CAMELYON16 report artifacts."""
    args = parse_args()
    output = Path(args.output_dir)
    selection_dir = Path(args.selection_dir)
    tables = output / "tables"
    figures = output / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    report = cast(
        dict, json.loads(Path(args.prepare_report).read_text(encoding="utf-8"))
    )
    write_dataset_summary(report, tables / "camelyon16_dataset_summary.tex")
    plot_distribution(report, figures / "camelyon16_class_distribution.png")
    results = result_summary(Path(args.results_dir), selection_dir)
    write_result_tables(results, tables)
    calibration = calibration_summary(Path(args.results_dir), selection_dir)
    write_calibration_tables(calibration, tables)


def write_dataset_summary(report: dict, path: Path) -> None:
    """Write a compact CAMELYON16 dataset summary table."""
    counts = cast(dict, report.get("class_counts_patch", {}))
    slide_counts = cast(dict, report.get("slide_counts", {}))
    ratio = float(report.get("imbalance_ratio_patch", 0.0))
    rows = [
        f"WSIs (tumor / normal) & \\num{{{int(slide_counts.get('tumor', 0))}}}"
        f" / \\num{{{int(slide_counts.get('normal', 0))}}}\\\\",
        f"Patch-labelled WSIs & \\num{{{int(report.get('n_patch_slides', 0))}}}\\\\",
        f"WSI-bag size (median tiles) & \\num{{{int(report.get('wsi_bag_size', 0))}}}\\\\",
        "Patch labels & tumor, normal\\\\",
        f"Tumor / normal patches & \\num{{{int(counts.get('tumor', 0))}}}"
        f" / \\num{{{int(counts.get('normal', 0))}}}\\\\",
        f"Patch imbalance ratio & ${{\\approx}}\\num{{{ratio:.1f}}}{{:}}1$\\\\",
    ]
    _write_table(path, "Property & Value", rows)


def plot_distribution(report: dict, path: Path) -> None:
    """Plot CAMELYON16 native patch support per tumor/normal label."""
    counts = cast(dict[str, int], report.get("class_counts_patch", {}))
    if not counts:
        return
    labels = list(LABELS)
    values = [int(counts.get(label, 0)) for label in labels]
    fig, ax = plt.subplots(figsize=(4.5, 3.6))
    ax.bar(labels, values, color="#4c78a8")
    ax.set_xlabel("CAMELYON16 patch label")
    ax.set_ylabel("Patches")
    ax.set_yscale("log")
    ax.set_title("Native CAMELYON16 patch support")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def result_summary(results_dir: Path, selection_dir: Path) -> pd.DataFrame:
    """Return native test results for tuning-selected winners."""
    rows = []
    for entry in _selection(selection_dir):
        for seed in range(3):
            path = _run_dir(results_dir, entry, seed) / "test_results.json"
            if not path.exists():
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "benchmark": "wsi_bag" if entry["benchmark"] == "wsi" else "patch",
                    "method": entry["method"],
                    "seed": seed,
                    "accuracy": float(payload["accuracy"]),
                    "balanced_accuracy": float(payload["balanced_accuracy"]),
                    "macro_f1": float(payload["macro_f1"]),
                }
            )
    return pd.DataFrame(rows)


def write_result_tables(frame: pd.DataFrame, tables: Path) -> None:
    """Write native CAMELYON16 patch and WSI result tables."""
    for benchmark, filename in (
        ("patch", "camelyon16_result_summary_patch.tex"),
        ("wsi_bag", "camelyon16_result_summary_wsi_bag.tex"),
    ):
        part = frame[frame["benchmark"] == benchmark] if not frame.empty else frame
        write_result_table(cast(pd.DataFrame, part), tables / filename)


def write_result_table(frame: pd.DataFrame, path: Path) -> None:
    """Write one native CAMELYON16 result table sorted by macro F1."""
    header = "Method & Accuracy & Balanced accuracy & Macro F1"
    if frame.empty:
        _write_unavailable(path, header)
        return
    stats = _result_stats(frame)
    rows = [
        " & ".join(
            [
                _method_label(row["method"]),
                _mean_std(row, "accuracy"),
                _mean_std(row, "balanced_accuracy"),
                _mean_std(row, "macro_f1"),
            ]
        )
        + "\\\\"
        for _, row in stats.iterrows()
    ]
    _write_table(path, header, rows)


def _result_stats(frame: pd.DataFrame) -> pd.DataFrame:
    stats = (
        frame.groupby("method")
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_std=("balanced_accuracy", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
        )
        .reset_index()
    )
    return stats.sort_values("macro_f1_mean", ascending=False, kind="stable")


def calibration_summary(results_dir: Path, selection_dir: Path) -> pd.DataFrame:
    """Return native CAMELYON16 calibration rows."""
    rows = []
    for entry in _selection(selection_dir):
        for seed in range(3):
            run = _run_dir(results_dir, entry, seed)
            val = run / "validation_results.json"
            test = run / "test_results.json"
            if not val.exists() or not test.exists():
                continue
            metadata = {
                "method": entry["method"],
                "order": "native",
                "parameter": "0",
                "seed": str(seed),
            }
            row = _calibration_row(metadata, val, test)
            if row:
                row["benchmark"] = _benchmark(entry["method"])
                rows.append(row)
    return pd.DataFrame(rows)


CALIBRATION_HEADER = "Method & ECE & ECE+TS & $T$"


def write_calibration_tables(frame: pd.DataFrame, tables: Path) -> None:
    """Write compact native ECE tables for CAMELYON16."""
    for benchmark, filename, order in (
        ("patch", "camelyon16_calibration_patch.tex", PATCH_ORDER),
        ("wsi_bag", "camelyon16_calibration_wsi_bag.tex", WSI_ORDER),
    ):
        part = frame[frame["benchmark"] == benchmark] if not frame.empty else frame
        _write_calibration_table(cast(pd.DataFrame, part), tables / filename, order)


def _write_calibration_table(part: pd.DataFrame, path: Path, order: list[str]) -> None:
    if part.empty:
        _write_unavailable(path, CALIBRATION_HEADER)
        return
    agg = _calibration_stats(part)
    methods = [m for m in order if m in set(agg["method"])] + [
        str(m) for m in agg["method"].tolist() if m not in order
    ]
    rows = [_calibration_row_tex(agg, method) for method in methods]
    _write_table(path, CALIBRATION_HEADER, rows)


def _calibration_stats(part: pd.DataFrame) -> pd.DataFrame:
    metrics = (
        "expected_calibration_error",
        "expected_calibration_error_scaled",
        "temperature",
    )
    agg = {f"{m}_{s}": (m, s) for m in metrics for s in ("mean", "std")}
    return part.groupby("method").agg(**agg).reset_index()


def _calibration_row_tex(agg: pd.DataFrame, method: str) -> str:
    row = agg[agg["method"] == method].iloc[0]
    return (
        " & ".join(
            [
                _method_label(method),
                _mean_std(row, "expected_calibration_error"),
                _mean_std(row, "expected_calibration_error_scaled"),
                _mean_std(row, "temperature"),
            ]
        )
        + "\\\\"
    )


def _selection(selection_dir: Path) -> list[dict]:
    path = selection_dir / "tuning_selection.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def _run_dir(results_dir: Path, entry: dict, seed: int) -> Path:
    stem = results_dir / "tuning" / entry["benchmark"] / "native"
    return stem / entry["method"] / entry["variant"] / f"seed={seed}"


if __name__ == "__main__":
    main()
