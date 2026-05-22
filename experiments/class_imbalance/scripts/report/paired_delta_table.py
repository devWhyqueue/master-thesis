from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from scripts.common import ensure_dirs, load_config

METRICS = ("macro_f1", "balanced_accuracy")
METRIC_HEADERS = {
    "macro_f1": r"$\Delta$ Macro F1",
    "balanced_accuracy": r"$\Delta$ Balanced accuracy",
}


@dataclass(frozen=True)
class PairedComparison:
    benchmark: str
    method: str
    baseline: str
    label: str


PAIRED_COMPARISONS = (
    PairedComparison(
        "patch",
        "patch_ce_soft_mcc_balanced",
        "patch_ce",
        r"CE + soft MCC $-$ CE",
    ),
    PairedComparison(
        "patch",
        "patch_ce_soft_f1_balanced",
        "patch_ce",
        r"CE + soft F1 $-$ CE",
    ),
    PairedComparison("wsi_bag", "rankmix_mil", "mil_ce", r"RankMix $-$ MIL CE"),
)


def parse_args() -> argparse.Namespace:
    """Parse paired-delta table generation arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    return parser.parse_args()


def _load_by_seed(paths: dict[str, Path], benchmark: str) -> pd.DataFrame:
    path = paths["tables"] / f"result_summary_{benchmark}_by_seed.csv"
    return pd.read_csv(path)


def _paired_metric_values(
    frame: pd.DataFrame,
    comparison: PairedComparison,
    metric: str,
    split: str,
) -> pd.Series:
    selected = frame[frame["split"] == split]
    rows: list[float] = []
    for seed in sorted(selected["seed"].unique()):
        method_row = selected[
            (selected["method"] == comparison.method) & (selected["seed"] == seed)
        ].iloc[0]
        baseline_row = selected[
            (selected["method"] == comparison.baseline) & (selected["seed"] == seed)
        ].iloc[0]
        rows.append(float(method_row[metric]) - float(baseline_row[metric]))
    return pd.Series(rows, dtype=float)


def _format_delta_cell(values: pd.Series) -> str:
    seed_values = ", ".join(f"{value:+.3f}" for value in values)
    mean = float(values.mean())
    std = float(values.std(ddof=0))
    return (
        f"${mean:+.3f} \\pm {std:.3f}$ {{\\scriptsize $[{seed_values}]$}}"
    )


def build_paired_delta_table(
    paths: dict[str, Path], split: str
) -> pd.DataFrame:
    """Build paired seed-difference rows for the headline comparisons."""
    cache: dict[str, pd.DataFrame] = {}
    rows: list[dict[str, str]] = []
    for comparison in PAIRED_COMPARISONS:
        if comparison.benchmark not in cache:
            cache[comparison.benchmark] = _load_by_seed(paths, comparison.benchmark)
        frame = cache[comparison.benchmark]
        row: dict[str, str] = {"comparison": comparison.label}
        for metric in METRICS:
            values = _paired_metric_values(frame, comparison, metric, split)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=0))
            row[f"{metric}_seed_values"] = ",".join(
                f"{value:+.6f}" for value in values
            )
            row[metric] = _format_delta_cell(values)
        rows.append(row)
    return pd.DataFrame(rows)


def _write_latex(frame: pd.DataFrame, path: Path) -> None:
    lines = [
        "\\begin{tabular}{lcc}",
        "\\toprule",
        f"Comparison & {METRIC_HEADERS['macro_f1']} & "
        f"{METRIC_HEADERS['balanced_accuracy']}\\\\",
        "\\midrule",
    ]
    for row in frame.to_dict("records"):
        lines.append(
            f"{row['comparison']} & {row['macro_f1']} & {row['balanced_accuracy']}\\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Write paired seed-difference table artifacts for the paper."""
    args = parse_args()
    config = load_config(args.config)
    paths = ensure_dirs(config)
    frame = build_paired_delta_table(paths, args.split)
    stem = f"result_paired_deltas_{args.split}"
    frame.to_csv(paths["tables"] / f"{stem}.csv", index=False)
    _write_latex(frame, paths["tables"] / f"{stem}.tex")


if __name__ == "__main__":
    main()
