"""Build report tables and figures for the native PANDA benchmark.

The WSI-bag regime grades biopsies by 6-class ISUP; the patch regime detects
binary cancer/benign tiles. A placeholder row is appended to the patch tables.
"""

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
from analysis.plotting.support.calibration import _calibration_row

WSI_LABELS = ("ISUP0", "ISUP1", "ISUP2", "ISUP3", "ISUP4", "ISUP5")
PATCH_LABELS = ("benign", "cancer")
PROGAN_METHOD = "patch_feature_progan_aug"


def parse_args() -> argparse.Namespace:
    """Parse PANDA report aggregation arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-report", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--selection-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    """Write native PANDA report artifacts."""
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
    write_dataset_summary(report, tables / "panda_dataset_summary.tex")
    plot_distribution(report, figures / "panda_class_distribution.png")
    results = result_summary(Path(args.results_dir), selection_dir)
    write_result_tables(results, tables)
    calibration = calibration_summary(Path(args.results_dir), selection_dir)
    write_calibration_tables(calibration, tables)


def write_dataset_summary(report: dict, path: Path) -> None:
    """Write a compact PANDA dataset summary table."""
    patch = cast(dict, report.get("class_counts_patch", {}))
    wsi_ratio = float(report.get("imbalance_ratio_wsi", 0.0))
    patch_ratio = float(report.get("imbalance_ratio_patch", 0.0))
    rows = [
        f"WSIs (biopsies) & \\num{{{int(report.get('n_slides', 0))}}}\\\\",
        f"Patch-labelled WSIs & \\num{{{int(report.get('n_patch_slides', 0))}}}\\\\",
        f"WSI-bag size (median tiles) & \\num{{{int(report.get('wsi_bag_size', 0))}}}\\\\",
        "WSI labels & ISUP 0--5\\\\",
        f"WSI imbalance ratio & ${{\\approx}}\\num{{{wsi_ratio:.1f}}}{{:}}1$\\\\",
        "Patch labels & benign, cancer\\\\",
        f"Benign / cancer patches & \\num{{{int(patch.get('benign', 0))}}}"
        f" / \\num{{{int(patch.get('cancer', 0))}}}\\\\",
        f"Patch imbalance ratio & ${{\\approx}}\\num{{{patch_ratio:.1f}}}{{:}}1$\\\\",
    ]
    _write_table(path, "Property & Value", rows)


def plot_distribution(report: dict, path: Path) -> None:
    """Plot PANDA native WSI support per ISUP grade."""
    counts = cast(dict[str, int], report.get("class_counts_wsi", {}))
    if not counts:
        return
    labels = list(WSI_LABELS)
    values = [int(counts.get(label, 0)) for label in labels]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.bar(labels, values, color="#4c78a8")
    ax.set_xlabel("PANDA ISUP grade")
    ax.set_ylabel("WSIs")
    ax.set_title("Native PANDA ISUP support")
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
    """Write native PANDA patch and WSI result tables."""
    for benchmark, filename in (
        ("patch", "panda_result_summary_patch.tex"),
        ("wsi_bag", "panda_result_summary_wsi_bag.tex"),
    ):
        part = frame[frame["benchmark"] == benchmark] if not frame.empty else frame
        write_result_table(cast(pd.DataFrame, part), tables / filename, benchmark)


def write_result_table(frame: pd.DataFrame, path: Path, benchmark: str) -> None:
    """Write one native PANDA result table sorted by macro F1."""
    header = "Method & Accuracy & Balanced accuracy & Macro F1"
    if frame.empty:
        _write_unavailable(path, header)
        return
    stats = _result_stats(frame)
    rows = [
        " & ".join(
            [
                _method_label(str(row["method"])),
                _mean_std(row, "accuracy"),
                _mean_std(row, "balanced_accuracy"),
                _mean_std(row, "macro_f1"),
            ]
        )
        + "\\\\"
        for _, row in stats.iterrows()
    ]
    if benchmark == "patch" and PROGAN_METHOD not in set(stats["method"]):
        rows.append(_placeholder_row())
    _write_table(path, header, rows)


def _agg_stats(frame: pd.DataFrame, metrics: tuple[str, ...]) -> pd.DataFrame:
    agg = {f"{m}_{s}": (m, s) for m in metrics for s in ("mean", "std")}
    return frame.groupby("method").agg(**agg).reset_index()


def _result_stats(frame: pd.DataFrame) -> pd.DataFrame:
    stats = _agg_stats(frame, ("accuracy", "balanced_accuracy", "macro_f1"))
    return stats.sort_values("macro_f1_mean", ascending=False, kind="stable")


def calibration_summary(results_dir: Path, selection_dir: Path) -> pd.DataFrame:
    """Return native PANDA calibration rows."""
    rows = []
    for entry in _selection(selection_dir):
        for seed in range(3):
            run = _run_dir(results_dir, entry, seed)
            val = run / "validation_results.json"
            test = run / "test_results.json"
            if not val.exists() or not test.exists():
                continue
            meta = {"method": entry["method"], "order": "native", "seed": str(seed)}
            row = _calibration_row({**meta, "parameter": "0"}, val, test)
            if row:
                row["benchmark"] = _benchmark(entry["method"])
                rows.append(row)
    return pd.DataFrame(rows)


CALIBRATION_HEADER = "Method & ECE & ECE+TS & $T$"
_CALIBRATION_METRICS = (
    "expected_calibration_error",
    "expected_calibration_error_scaled",
    "temperature",
)


def write_calibration_tables(frame: pd.DataFrame, tables: Path) -> None:
    """Write compact native ECE tables for PANDA."""
    for benchmark, filename, order in (
        ("patch", "panda_calibration_patch.tex", PATCH_ORDER),
        ("wsi_bag", "panda_calibration_wsi_bag.tex", WSI_ORDER),
    ):
        part = frame[frame["benchmark"] == benchmark] if not frame.empty else frame
        _write_calibration_table(
            cast(pd.DataFrame, part), tables / filename, order, benchmark
        )


def _write_calibration_table(
    part: pd.DataFrame, path: Path, order: list[str], benchmark: str
) -> None:
    if part.empty:
        _write_unavailable(path, CALIBRATION_HEADER)
        return
    agg = _agg_stats(part, _CALIBRATION_METRICS)
    methods = [m for m in order if m in set(agg["method"])] + [
        str(m) for m in agg["method"].tolist() if m not in order
    ]
    rows = [
        " & ".join(
            [_method_label(m)]
            + [
                _mean_std(agg[agg["method"] == m].iloc[0], k)
                for k in _CALIBRATION_METRICS
            ]
        )
        + "\\\\"
        for m in methods
    ]
    if benchmark == "patch" and PROGAN_METHOD not in set(agg["method"]):
        rows.append(_placeholder_row())
    _write_table(path, CALIBRATION_HEADER, rows)


def _placeholder_row() -> str:
    """Return the excluded-ProGAN placeholder row (four columns, em-dashes)."""
    return f"{_method_label(PROGAN_METHOD)} & -- & -- & --\\\\"


def _selection(selection_dir: Path) -> list[dict]:
    path = selection_dir / "tuning_selection.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def _run_dir(results_dir: Path, entry: dict, seed: int) -> Path:
    stem = results_dir / "tuning" / entry["benchmark"] / "native"
    return stem / entry["method"] / entry["variant"] / f"seed={seed}"


if __name__ == "__main__":
    main()
