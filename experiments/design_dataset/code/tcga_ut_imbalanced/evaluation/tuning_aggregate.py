"""Aggregate per-regime validation tuning and select winners by macro-F1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import pandas as pd

from tcga_ut_imbalanced.evaluation.tuning_grid import patch_grid, regimes, wsi_grid


def parse_args() -> argparse.Namespace:
    """Parse tuning aggregation arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Select tuning winners and write report artifacts."""
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _collect_rows(results_dir)
    if rows.empty:
        (output_dir / "tuning_selection.json").write_text("[]\n", encoding="utf-8")
        return
    selected = _select_winners(rows, args.allow_incomplete)
    (output_dir / "tuning_selection.json").write_text(
        json.dumps(selected, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_latex(selected, output_dir / "result_tuning_selection.tex")


def _collect_rows(results_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for benchmark, grid in (("patch", patch_grid()), ("wsi", wsi_grid())):
        for regime in regimes():
            for variant in grid:
                for seed in (0, 1, 2):
                    result_dir = (
                        results_dir
                        / "tuning"
                        / benchmark
                        / regime.label
                        / variant.method
                        / variant.variant
                        / f"seed={seed}"
                    )
                    val_path = result_dir / "validation_results.json"
                    test_path = result_dir / "test_results.json"
                    if not val_path.exists() or not test_path.exists():
                        continue
                    val = json.loads(val_path.read_text(encoding="utf-8"))
                    test = json.loads(test_path.read_text(encoding="utf-8"))
                    rows.append(
                        {
                            "benchmark": benchmark,
                            "regime": regime.label,
                            "method": variant.method,
                            "variant": variant.variant,
                            "params": variant.params_json,
                            "seed": seed,
                            "val_macro_f1": float(val["macro_f1"]),
                            "val_balanced_accuracy": float(val["balanced_accuracy"]),
                            "test_macro_f1": float(test["macro_f1"]),
                            "test_balanced_accuracy": float(test["balanced_accuracy"]),
                        }
                    )
    return pd.DataFrame(rows)


def _select_winners(
    frame: pd.DataFrame, allow_incomplete: bool
) -> list[dict[str, Any]]:
    grouped = frame.groupby(
        ["benchmark", "regime", "method", "variant", "params"], as_index=False
    ).agg(
        val_macro_f1_mean=("val_macro_f1", "mean"),
        val_balanced_accuracy_mean=("val_balanced_accuracy", "mean"),
        test_macro_f1_mean=("test_macro_f1", "mean"),
        test_balanced_accuracy_mean=("test_balanced_accuracy", "mean"),
        n_seeds=("seed", "count"),
    )
    selected: list[dict[str, Any]] = []
    for key, group in grouped.groupby(["benchmark", "regime", "method"], sort=False):
        benchmark, regime, method = cast(tuple[str, str, str], key)
        if not allow_incomplete and int(group["n_seeds"].max()) != 3:
            raise ValueError(f"Incomplete tuning for {benchmark}/{regime}/{method}")
        winner = max(
            (row for _, row in group.iterrows()),
            key=lambda row: (
                float(row["val_macro_f1_mean"]),
                float(row["val_balanced_accuracy_mean"]),
            ),
        )
        selected.append(
            {
                "benchmark": benchmark,
                "regime": regime,
                "method": method,
                "variant": winner["variant"],
                "selected_params": winner["params"],
                "val_macro_f1": float(winner["val_macro_f1_mean"]),
                "test_macro_f1": float(winner["test_macro_f1_mean"]),
                "test_balanced_accuracy": float(winner["test_balanced_accuracy_mean"]),
            }
        )
    return selected


def _write_latex(rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "\\begin{tabular}{llllrr}",
        "\\toprule",
        "Benchmark & Regime & Method & Variant & Val macro-F1 & Test macro-F1\\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['benchmark']} & {row['regime']} & {row['method']} & "
            f"{row['variant']} & {row['val_macro_f1']:.3f} & "
            f"{row['test_macro_f1']:.3f}\\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
