import argparse
import json
from pathlib import Path
from typing import cast

import matplotlib.axes
import matplotlib.pyplot as plt
import matplotlib.ticker
import pandas as pd

from analysis.evaluation.tuning_grid import CLASS_ORDERS, LAMBDAS
from analysis.plotting import (
    SEVERITY_COLORS,
    SPLIT_PATTERN,
    _write_table,
    _write_unavailable,
)
from analysis.plotting.support.calibration import (
    calibration_summary,
    write_calibration_tables,
)
from analysis.plotting.results import (
    native_results,
    result_summary,
    write_result_tables,
)
from analysis.plotting.support.tail_class import (
    plot_support_vs_recall,
    tail_class_frame,
    write_tail_class_tables,
)


def get_args() -> argparse.Namespace:
    """Parse report aggregation arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--constructed-dataset-dir", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--no-native-reference", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Build constructed-report tables and figures."""
    args = get_args()
    output_dir = Path(args.output_dir)
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    constructed_dir = Path(args.constructed_dataset_dir)
    split_frame = split_summary(constructed_dir)
    native = native_distribution(constructed_dir)
    write_split_summary(
        split_frame, native, tables_dir / "constructed_split_summary.tex"
    )
    plot_support(split_frame, native, figures_dir / "constructed_support_by_rank.png")
    result_frame = result_summary(Path(args.results_dir), output_dir)
    native_reference = None if args.no_native_reference else native_results()
    write_result_tables(result_frame, tables_dir, native_reference)
    calibration_frame = calibration_summary(Path(args.results_dir), output_dir)
    write_calibration_tables(calibration_frame, tables_dir)
    tail_frame = tail_class_frame(
        Path(args.results_dir), output_dir, constructed_dir, native
    )
    write_tail_class_tables(tail_frame, tables_dir)
    plot_support_vs_recall(tail_frame, figures_dir / "support_vs_recall.png")


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


SPLIT_HEADER = "$\\lambda$ & Training slides & Support range & Head:tail ratio"


def write_split_summary(frame: pd.DataFrame, native: pd.Series, path: Path) -> None:
    """Write the constructed split summary table, including the native full-pool row."""
    if frame.empty:
        _write_unavailable(path, SPLIT_HEADER)
        return
    rows = _split_summary_rows(frame)
    if not native.empty:
        total, lo, hi = float(native.sum()), float(native.min()), float(native.max())
        rows.append(_support_row("native (full pool)", total, lo, hi))
    _write_table(path, SPLIT_HEADER, rows)


def _split_summary_rows(frame: pd.DataFrame) -> list[str]:
    agg = (
        frame.groupby(["parameter", "seed"])
        .agg(total=("count", "sum"), minimum=("count", "min"), maximum=("count", "max"))
        .groupby("parameter")
        .mean()
        .reset_index()
    )
    return [
        _support_row(f"{r['parameter']:.1f}", r["total"], r["minimum"], r["maximum"])
        for r in agg.to_dict("records")
    ]


def _support_row(label: str, total: float, minimum: float, maximum: float) -> str:
    ratio = round(maximum / minimum) if minimum else 0
    return (
        f"{label} & {total:.0f} & {minimum:.0f}--{maximum:.0f} & "
        f"${{\\approx}}{ratio}{{:}}1$\\\\"
    )


def plot_support(frame: pd.DataFrame, native: pd.Series, path: Path) -> None:
    """Plot training support by rank for the three evaluated severities plus native reference."""
    if frame.empty:
        return
    grouped = (
        frame[frame["order"] == "native_prevalence"]
        .groupby(["parameter", "rank"])["count"]
        .mean()
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    _add_severity_lines(ax, grouped)
    _add_native_line(ax, native)
    ax.set_xlabel("Class rank")
    ax.set_ylabel("Training slides")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _add_native_line(ax: matplotlib.axes.Axes, native: pd.Series) -> None:
    if native.empty:
        return
    ax.plot(
        range(1, len(native) + 1),
        native.to_numpy(),
        color="black",
        linestyle="--",
        label="Native (full pool)",
    )


def native_distribution(root: Path) -> pd.Series:
    """Return seed-averaged native full-pool counts ordered by native-prevalence rank."""
    by_seed: dict[str, dict[str, int]] = {}
    for path in sorted(root.glob("constructed_order=native_prevalence_*_seed=*")):
        match = SPLIT_PATTERN.fullmatch(path.name)
        counts_path = path / "available_counts.json"
        if match is None or not counts_path.exists():
            continue
        by_seed[match.group("seed")] = json.loads(
            counts_path.read_text(encoding="utf-8")
        )
    if not by_seed:
        return pd.Series(dtype=float)
    mean = cast(pd.Series, pd.DataFrame(by_seed.values()).mean())
    return cast(pd.Series, mean.sort_values(ascending=False))


def _add_severity_lines(ax: matplotlib.axes.Axes, grouped: pd.DataFrame) -> None:
    params = [0.5, 1.0, 1.5]
    for param, color in zip(params, SEVERITY_COLORS):
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
            "count": int(count),
        }
        for rank, (_class_name, count) in enumerate(ordered, start=1)
    ]


if __name__ == "__main__":
    main()
