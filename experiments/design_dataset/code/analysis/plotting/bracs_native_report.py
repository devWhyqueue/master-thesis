"""Build report tables and figures for the native BRACS benchmark."""

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

LABELS = ("N", "PB", "UDH", "FEA", "ADH", "DCIS", "IC")


def parse_args() -> argparse.Namespace:
    """Parse BRACS report aggregation arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-root", required=True)
    parser.add_argument("--prepare-report", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    """Write native BRACS report artifacts."""
    args = parse_args()
    output = Path(args.output_dir)
    tables = output / "tables"
    figures = output / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    report_path = Path(args.prepare_report)
    report = cast(dict, json.loads(report_path.read_text(encoding="utf-8")))
    write_dataset_summary(report, tables / "bracs_dataset_summary.tex")
    plot_distribution(report, figures / "bracs_class_distribution.png")
    result_frame = result_summary(Path(args.results_dir), output)
    write_result_tables(result_frame, tables)
    calibration_frame = calibration_summary(Path(args.results_dir), output)
    write_calibration_tables(calibration_frame, tables)


def write_dataset_summary(report: dict, path: Path) -> None:
    """Write a compact BRACS dataset summary table."""
    counts = cast(dict, report.get("class_counts_wsi", {})).values()
    ratio = float(report.get("imbalance_ratio_wsi", 0.0))
    rows = [
        f"Patients & \\num{{{int(report.get('n_patients', 0))}}}\\\\",
        f"WSIs with ROI tiles & \\num{{{int(report.get('n_wsis_with_tiles', 0))}}}\\\\",
        f"ROI metadata rows & \\num{{{int(report.get('n_roi_metadata_rows', 0))}}}\\\\",
        f"ROI tiles & \\num{{{int(report.get('n_tiled_rows', 0))}}}\\\\",
        f"WSI-bag size (median tiles) & \\num{{{int(report.get('wsi_bag_size', 0))}}}\\\\",
        f"Subtype labels & {', '.join(LABELS)}\\\\",
        f"WSI support range & \\num{{{min(counts, default=0)}}}--\\num{{{max(counts, default=0)}}}\\\\",
        f"WSI imbalance ratio & ${{\\approx}}\\num{{{ratio:.1f}}}{{:}}1$\\\\",
    ]
    _write_table(path, "Property & Value", rows)


def plot_distribution(report: dict, path: Path) -> None:
    """Plot BRACS native WSI support per subtype."""
    counts = cast(dict[str, int], report.get("class_counts_wsi", {}))
    if not counts:
        return
    labels = list(LABELS)
    values = [int(counts.get(label, 0)) for label in labels]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.bar(labels, values, color="#4c78a8")
    ax.set_xlabel("BRACS subtype")
    ax.set_ylabel("WSIs")
    ax.set_title("Native BRACS subtype support")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def result_summary(results_dir: Path, output_dir: Path) -> pd.DataFrame:
    """Return native test results for tuning-selected winners."""
    selection = _selection(output_dir)
    rows = []
    for entry in selection:
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
    """Write native BRACS patch and WSI result tables."""
    for benchmark, filename in (
        ("patch", "bracs_result_summary_patch.tex"),
        ("wsi_bag", "bracs_result_summary_wsi_bag.tex"),
    ):
        part = frame[frame["benchmark"] == benchmark] if not frame.empty else frame
        write_result_table(cast(pd.DataFrame, part), tables / filename)


def write_result_table(frame: pd.DataFrame, path: Path) -> None:
    """Write one native BRACS result table."""
    header = "Method & Accuracy & Balanced accuracy & Macro F1"
    if frame.empty:
        _write_unavailable(path, header)
        return
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
    stats = stats.sort_values("macro_f1_mean", ascending=False, kind="stable")
    rows = []
    for _, row in stats.iterrows():
        rows.append(
            " & ".join(
                [
                    _method_label(row["method"]),
                    _mean_std(row, "accuracy"),
                    _mean_std(row, "balanced_accuracy"),
                    _mean_std(row, "macro_f1"),
                ]
            )
            + "\\\\"
        )
    _write_table(path, header, rows)


def calibration_summary(results_dir: Path, output_dir: Path) -> pd.DataFrame:
    """Return native BRACS calibration rows."""
    rows = []
    for entry in _selection(output_dir):
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
            row = _calibration_row(
                metadata,
                val,
                test,
            )
            if row:
                row["benchmark"] = _benchmark(entry["method"])
                rows.append(row)
    return pd.DataFrame(rows)


def write_calibration_tables(frame: pd.DataFrame, tables: Path) -> None:
    """Write compact native ECE tables for BRACS."""
    header = "Method & ECE & ECE+TS & $T$"
    for benchmark, filename, order in (
        ("patch", "bracs_calibration_patch.tex", PATCH_ORDER),
        ("wsi_bag", "bracs_calibration_wsi_bag.tex", WSI_ORDER),
    ):
        part = frame[frame["benchmark"] == benchmark] if not frame.empty else frame
        path = tables / filename
        if part.empty:
            _write_unavailable(path, header)
            continue
        agg = (
            part.groupby("method")
            .agg(
                expected_calibration_error_mean=("expected_calibration_error", "mean"),
                expected_calibration_error_std=("expected_calibration_error", "std"),
                expected_calibration_error_scaled_mean=(
                    "expected_calibration_error_scaled",
                    "mean",
                ),
                expected_calibration_error_scaled_std=(
                    "expected_calibration_error_scaled",
                    "std",
                ),
                temperature_mean=("temperature", "mean"),
                temperature_std=("temperature", "std"),
            )
            .reset_index()
        )
        methods = [m for m in order if m in set(agg["method"])] + [
            str(m) for m in agg["method"].tolist() if m not in order
        ]
        rows = [
            " & ".join(
                [
                    _method_label(method),
                    _mean_std(
                        agg[agg["method"] == method].iloc[0],
                        "expected_calibration_error",
                    ),
                    _mean_std(
                        agg[agg["method"] == method].iloc[0],
                        "expected_calibration_error_scaled",
                    ),
                    _mean_std(agg[agg["method"] == method].iloc[0], "temperature"),
                ]
            )
            + "\\\\"
            for method in methods
        ]
        _write_table(path, header, rows)


def _selection(output_dir: Path) -> list[dict]:
    path = output_dir / "tuning_selection.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def _run_dir(results_dir: Path, entry: dict, seed: int) -> Path:
    stem = results_dir / "tuning" / entry["benchmark"] / "native"
    return stem / entry["method"] / entry["variant"] / f"seed={seed}"


if __name__ == "__main__":
    main()
