"""Aggregate native-dataset validation tuning and select winners."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import pandas as pd

from analysis.evaluation.native_tuning_grid import patch_grid, wsi_grid


def parse_args() -> argparse.Namespace:
    """Parse native tuning aggregation arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Select native tuning winners and write report artifacts."""
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = collect_rows(Path(args.results_dir))
    selected = [] if rows.empty else select_winners(rows, bool(args.allow_incomplete))
    (output / "tuning_selection.json").write_text(
        json.dumps(selected, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def collect_rows(results_dir: Path) -> pd.DataFrame:
    """Collect native validation/test rows."""
    rows: list[dict[str, Any]] = []
    for benchmark, grid in (("patch", patch_grid()), ("wsi", wsi_grid())):
        for variant in grid:
            for seed in (0, 1, 2):
                run = (
                    results_dir
                    / "tuning"
                    / benchmark
                    / "native"
                    / variant.method
                    / variant.variant
                    / f"seed={seed}"
                )
                val_path = run / "validation_results.json"
                test_path = run / "test_results.json"
                if not val_path.exists() or not test_path.exists():
                    continue
                val = json.loads(val_path.read_text(encoding="utf-8"))
                test = json.loads(test_path.read_text(encoding="utf-8"))
                rows.append(
                    {
                        "benchmark": benchmark,
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


def select_winners(frame: pd.DataFrame, allow_incomplete: bool) -> list[dict[str, Any]]:
    """Select one native variant per benchmark and method."""
    grouped = frame.groupby(
        ["benchmark", "method", "variant", "params"], as_index=False
    ).agg(
        val_macro_f1_mean=("val_macro_f1", "mean"),
        val_balanced_accuracy_mean=("val_balanced_accuracy", "mean"),
        test_macro_f1_mean=("test_macro_f1", "mean"),
        test_balanced_accuracy_mean=("test_balanced_accuracy", "mean"),
        n_seeds=("seed", "count"),
    )
    selected = []
    for key, group in grouped.groupby(["benchmark", "method"], sort=False):
        benchmark, method = cast(tuple[str, str], key)
        if not allow_incomplete and int(group["n_seeds"].max()) != 3:
            raise ValueError(f"Incomplete native tuning for {benchmark}/{method}")
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
                "regime": "native",
                "method": method,
                "variant": winner["variant"],
                "selected_params": winner["params"],
                "val_macro_f1": float(winner["val_macro_f1_mean"]),
                "test_macro_f1": float(winner["test_macro_f1_mean"]),
                "test_balanced_accuracy": float(winner["test_balanced_accuracy_mean"]),
            }
        )
    return selected


if __name__ == "__main__":
    main()
