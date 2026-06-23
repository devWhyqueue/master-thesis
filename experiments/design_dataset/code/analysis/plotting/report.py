import argparse
import json
import re
from pathlib import Path
from typing import cast

import matplotlib.axes
import matplotlib.pyplot as plt
import matplotlib.ticker
import pandas as pd

from analysis.evaluation.tuning_grid import CLASS_ORDERS, LAMBDAS
from analysis.plotting import (
    _benchmark,
    _tex,
    _write_table,
    _write_unavailable,
    write_result_tables,
)
from analysis.plotting.calibration import calibration_summary, write_calibration_tables

SPLIT_PATTERN = re.compile(
    r"constructed_order=(?P<order>.+)_parameter=(?P<parameter>[\d.]+)_seed=(?P<seed>\d+)"
)
METRICS = ("accuracy", "balanced_accuracy", "macro_f1")


def get_args() -> argparse.Namespace:
    """Parse report aggregation arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--constructed-dataset-dir", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    """Build constructed-report tables and figures."""
    args = get_args()
    output_dir = Path(args.output_dir)
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    split_frame = split_summary(Path(args.constructed_dataset_dir))
    write_split_summary(split_frame, tables_dir / "constructed_split_summary.tex")
    plot_support(split_frame, figures_dir / "constructed_support_by_rank.png")
    result_frame = result_summary(Path(args.results_dir), output_dir)
    write_result_tables(result_frame, tables_dir)
    calibration_frame = calibration_summary(Path(args.results_dir), output_dir)
    write_calibration_tables(calibration_frame, tables_dir)


def split_summary(root: Path) -> pd.DataFrame:
    """Return one row per constructed split and class."""
    rows = []
    for path in sorted(root.glob("constructed_order=*_parameter=*_seed=*")):
        match = SPLIT_PATTERN.fullmatch(path.name)
        counts_path = path / "target_counts.json"
        if match is None or not counts_path.exists():
            continue
        if match.group("order") not in CLASS_ORDERS:
            continue
        if float(match.group("parameter")) not in LAMBDAS:
            continue
        rows.extend(_split_rows(match.groupdict(), counts_path))
    return pd.DataFrame(rows)


def result_summary(results_dir: Path, output_dir: Path) -> pd.DataFrame:
    """Return test results for tuning-selected winners, one row per (method, regime, seed)."""
    selection_path = output_dir / "tuning_selection.json"
    if not selection_path.exists():
        return pd.DataFrame()
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if not selection:
        return pd.DataFrame()
    return pd.DataFrame(_result_rows(results_dir, selection))


def _result_rows(results_dir: Path, selection: list[dict]) -> list[dict]:
    return [
        row for entry in selection for row in _entry_result_rows(entry, results_dir)
    ]


def _entry_result_rows(entry: dict, results_dir: Path) -> list[dict]:
    m = re.match(r"order=(?P<order>.+)/param=(?P<parameter>[\d.]+)", entry["regime"])
    if m is None:
        return []
    order, parameter = m.group("order"), float(m.group("parameter"))
    rows = []
    for seed in range(3):
        test_path = (
            results_dir
            / "tuning"
            / entry["benchmark"]
            / entry["regime"]
            / entry["method"]
            / entry["variant"]
            / f"seed={seed}"
            / "test_results.json"
        )
        row = _load_result_row(test_path, entry, order, parameter, seed)
        if row:
            rows.append(row)
    return rows


def _load_result_row(
    test_path: Path, entry: dict, order: str, parameter: float, seed: int
) -> dict | None:
    if not test_path.exists():
        return None
    with open(test_path, encoding="utf-8") as f:
        payload = json.load(f)
    return {
        "method": entry["method"],
        "benchmark": _benchmark(entry["method"]),
        "order": order,
        "parameter": parameter,
        "seed": seed,
        **{metric: float(payload[metric]) for metric in METRICS},
    }


def write_split_summary(frame: pd.DataFrame, path: Path) -> None:
    """Write the constructed split summary table."""
    if frame.empty:
        _write_unavailable(
            path, "Order & $\\lambda$ & Training slides & Minimum class support"
        )
        return
    rows = _split_summary_rows(frame)
    _write_table(path, "Order & $\\lambda$ & Training slides & Support range", rows)


def _split_summary_rows(frame: pd.DataFrame) -> list[str]:
    per_seed = (
        frame.groupby(["order", "parameter", "seed"])
        .agg(total=("count", "sum"), minimum=("count", "min"), maximum=("count", "max"))
        .reset_index()
    )
    aggregate = (
        per_seed.groupby(["order", "parameter"])
        .agg(
            total=("total", "mean"),
            minimum=("minimum", "mean"),
            maximum=("maximum", "mean"),
        )
        .reset_index()
    )
    return [
        f"{_tex(row['order'])} & {float(row['parameter']):.1f} & "
        f"{float(row['total']):.0f} & "
        f"{float(row['minimum']):.1f}--{float(row['maximum']):.1f}\\\\"
        for _, row in aggregate.iterrows()
    ]


def plot_support(frame: pd.DataFrame, path: Path) -> None:
    """Plot training support by rank for the three evaluated severities plus native reference."""
    if frame.empty:
        return
    native_order = "native_prevalence"
    # ponytail: native full-pool split (largest total) reused as the native reference curve
    native_param = _find_native_param(frame, native_order)
    grouped = (
        frame[frame["order"] == native_order]
        .groupby(["parameter", "rank"])["count"]
        .mean()
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    _add_severity_lines(ax, grouped)
    _add_native_line(ax, grouped, native_param)
    ax.set_xlabel("Class rank")
    ax.set_ylabel("Training slides")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _add_native_line(
    ax: matplotlib.axes.Axes, grouped: pd.DataFrame, native_param: float
) -> None:
    native_part = grouped[grouped["parameter"] == native_param]
    if not native_part.empty:
        ax.plot(
            native_part["rank"],
            native_part["count"],
            color="black",
            linestyle="--",
            label="Native (full pool)",
        )


def _find_native_param(frame: pd.DataFrame, order: str) -> float:
    totals = (
        frame[frame["order"] == order].groupby("parameter")["count"].sum().reset_index()
    )
    return float(totals.loc[cast(int, totals["count"].idxmax()), "parameter"])


def _add_severity_lines(ax: matplotlib.axes.Axes, grouped: pd.DataFrame) -> None:
    params = [0.5, 1.0, 1.5]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for param, color in zip(params, colors):
        part = grouped[grouped["parameter"] == param]
        if not part.empty:
            ax.plot(
                part["rank"],
                part["count"],
                color=color,
                label=f"$\\lambda={param:.1f}$",
            )


def _split_rows(metadata: dict[str, str], counts_path: Path) -> list[dict[str, object]]:
    with open(counts_path) as file:
        counts = json.load(file)
    ordered = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    return [
        {
            "order": metadata["order"],
            "parameter": float(metadata["parameter"]),
            "seed": int(metadata["seed"]),
            "rank": rank,
            "class_name": class_name,
            "count": int(count),
        }
        for rank, (class_name, count) in enumerate(ordered, start=1)
    ]


if __name__ == "__main__":
    main()
