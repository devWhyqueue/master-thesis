import argparse
import json
import re
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import pandas as pd

from analysis.plotting import (
    _benchmark,
    _mean_std,
    _tex,
    _write_table,
    _write_unavailable,
)
from analysis.plotting.calibration import (
    calibration_summary,
    write_calibration_tables,
)

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
    return pd.DataFrame(_result_runs(results_dir, selection))


def _result_runs(results_dir: Path, selection: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for entry in selection:
        rows.extend(_entry_result_rows(entry, results_dir))
    return rows


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
        if not test_path.exists():
            continue
        with open(test_path, encoding="utf-8") as f:
            payload = json.load(f)
        rows.append(
            {
                "method": entry["method"],
                "benchmark": _benchmark(entry["method"]),
                "order": order,
                "parameter": parameter,
                "seed": seed,
                **{metric: float(payload[metric]) for metric in METRICS},
            }
        )
    return rows


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
    aggregate = _aggregate_split_summary(_per_seed_split_summary(frame))
    rows = []
    for _, row in aggregate.iterrows():
        rows.append(
            f"{_tex(row['order'])} & {float(row['parameter']):.1f} & "
            f"{float(row['total']):.0f} & "
            f"{float(row['minimum']):.1f}--{float(row['maximum']):.1f}\\\\"
        )
    return rows


def _per_seed_split_summary(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(["order", "parameter", "seed"])
        .agg(total=("count", "sum"), minimum=("count", "min"), maximum=("count", "max"))
        .reset_index()
    )


def _aggregate_split_summary(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(["order", "parameter"])
        .agg(
            total=("total", "mean"),
            minimum=("minimum", "mean"),
            maximum=("maximum", "mean"),
        )
        .reset_index()
    )


def plot_support(frame: pd.DataFrame, path: Path) -> None:
    """Plot constructed support by rank."""
    if frame.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    grouped = (
        frame.groupby(["order", "parameter", "rank"])["count"].mean().reset_index()
    )
    for key, part in grouped.groupby(["order", "parameter"]):
        order, parameter = cast(tuple[str, float], key)
        if float(parameter) not in {0.0, 1.0, 1.3}:
            continue
        ax.plot(
            part["rank"], part["count"], label=f"{order}, $\\lambda={parameter:.1f}$"
        )
    ax.set_xlabel("Class rank")
    ax.set_ylabel("Training slides")
    ax.set_yscale("log")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def write_result_tables(frame: pd.DataFrame, tables_dir: Path) -> None:
    """Write patch and WSI result summary tables."""
    for benchmark, filename in [
        ("patch", "result_summary_patch.tex"),
        ("wsi_bag", "result_summary_wsi_bag.tex"),
    ]:
        part = (
            cast(pd.DataFrame, frame[frame["benchmark"] == benchmark])
            if not frame.empty
            else frame
        )
        write_result_table(part, tables_dir / filename)


def write_result_table(frame: pd.DataFrame, path: Path) -> None:
    """Write one benchmark result table."""
    if frame.empty:
        _write_unavailable(path, "Method & Accuracy & Balanced accuracy & Macro F1")
        return
    aggregate = frame.groupby(["method", "order", "parameter"]).agg(
        **{f"{metric}_mean": (metric, "mean") for metric in METRICS},
        **{f"{metric}_std": (metric, "std") for metric in METRICS},
    )
    rows = []
    for _, row in aggregate.reset_index().iterrows():
        rows.append(
            f"{_tex(row['method'])} ({_tex(row['order'])}, "
            f"$\\lambda={float(row['parameter']):.1f}$) & "
            f"{_mean_std(row, 'accuracy')} & "
            f"{_mean_std(row, 'balanced_accuracy')} & "
            f"{_mean_std(row, 'macro_f1')}\\\\"
        )
    _write_table(path, "Method & Accuracy & Balanced accuracy & Macro F1", rows)


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
