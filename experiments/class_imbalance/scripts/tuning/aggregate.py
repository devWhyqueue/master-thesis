from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import pandas as pd

from scripts.common import ensure_dirs, load_config
from scripts.metadata import PATCH_FEATURE_METHOD_ALIASES
from scripts.report.figures import METHOD_LABELS
from scripts.tuning.grid import TuningVariant, grid_for_benchmark
from scripts.tuning.reporting import (
    write_delta_figure,
    write_empty_outputs,
    write_latex_table,
)


METRICS = ("accuracy", "balanced_accuracy", "macro_f1")
BASELINE_METHOD = {"patch_feature": "patch_feature_ce", "wsi_bag": "mil_ce"}
SUMMARY_STEM = {"patch_feature": "patch", "wsi_bag": "wsi_bag"}
REGIME_LABEL = {"patch_feature": "Patch", "wsi_bag": "WSI bag"}


def parse_args() -> argparse.Namespace:
    """Parse tuning aggregation arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Aggregate validation-tuning runs and write report artifacts."""
    args = parse_args()
    paths = ensure_dirs(load_config(args.config))
    frame = _collect_all(paths)
    if frame.empty:
        write_empty_outputs(paths)
        return
    selected = _select_all(frame, args.allow_incomplete)
    selected.to_csv(paths["tables"] / "result_tuning_selection.csv", index=False)
    write_latex_table(selected, paths["tables"] / "result_tuning_selection.tex")
    write_delta_figure(selected, paths["figures"] / "tuned_macro_f1_delta_test.png")


def _collect_all(paths: dict[str, Path]) -> pd.DataFrame:
    rows = []
    for benchmark in ("patch_feature", "wsi_bag"):
        fixed = _fixed_summary(paths, benchmark)
        for variant in grid_for_benchmark(benchmark):
            rows.extend(_variant_rows(paths, benchmark, variant, fixed))
    return pd.DataFrame(rows)


def _variant_rows(
    paths: dict[str, Path], benchmark: str, variant: TuningVariant, fixed: pd.DataFrame
) -> list[dict[str, Any]]:
    rows = []
    for seed in (0, 1, 2):
        result_dir = (
            paths["root"]
            / "outputs"
            / "tuning"
            / benchmark
            / variant.method
            / variant.variant
            / f"seed={seed}"
        )
        val = _read_result(result_dir / "val_results.json")
        test = _read_result(result_dir / "test_results.json")
        if val is None or test is None:
            continue
        rows.append(
            {
                "benchmark": benchmark,
                "method": variant.method,
                "method_label": _method_label(variant.method),
                "variant": variant.variant,
                "params": variant.params_json,
                "seed": seed,
                "fixed_test_macro_f1": _fixed_metric(fixed, variant.method, "macro_f1"),
                "fixed_test_balanced_accuracy": _fixed_metric(
                    fixed, variant.method, "balanced_accuracy"
                ),
                "baseline_test_macro_f1": _fixed_metric(
                    fixed, BASELINE_METHOD[benchmark], "macro_f1"
                ),
                **_prefixed_metrics("val", val),
                **_prefixed_metrics("test", test),
            }
        )
    return rows


def _select_all(frame: pd.DataFrame, allow_incomplete: bool) -> pd.DataFrame:
    selected_rows = []
    for key, group in frame.groupby(["benchmark", "method", "variant"], sort=False):
        benchmark, method, variant = cast(tuple[str, str, str], key)
        if not allow_incomplete and len(group) != 3:
            raise ValueError(
                f"Incomplete tuning results for {benchmark}/{method}/{variant}"
            )
    grouped = frame.groupby(
        ["benchmark", "method", "variant", "params"], as_index=False
    ).agg(
        method_label=("method_label", "first"),
        baseline_test_macro_f1=("baseline_test_macro_f1", "first"),
        fixed_test_macro_f1=("fixed_test_macro_f1", "first"),
        fixed_test_balanced_accuracy=("fixed_test_balanced_accuracy", "first"),
        val_macro_f1_mean=("val_macro_f1", "mean"),
        val_balanced_accuracy_mean=("val_balanced_accuracy", "mean"),
        test_macro_f1_mean=("test_macro_f1", "mean"),
        test_balanced_accuracy_mean=("test_balanced_accuracy", "mean"),
        test_accuracy_mean=("test_accuracy", "mean"),
    )
    for key, group in grouped.groupby(["benchmark", "method"], sort=False):
        benchmark, method = cast(tuple[str, str], key)
        winner = max(
            (row for _, row in group.iterrows()),
            key=lambda row: (
                float(row["val_macro_f1_mean"]),
                float(row["val_balanced_accuracy_mean"]),
            ),
        )
        selected_rows.append(_selection_row(benchmark, method, winner))
    return _with_baseline_rows(pd.DataFrame(selected_rows))


def _selection_row(benchmark: str, method: str, row: pd.Series) -> dict[str, Any]:
    baseline = BASELINE_METHOD[benchmark]
    baseline_macro_f1 = float(row["baseline_test_macro_f1"])
    return {
        "benchmark": benchmark,
        "regime": REGIME_LABEL[benchmark],
        "method": method,
        "method_label": row["method_label"],
        "selected_params": row["params"],
        "baseline_method": baseline,
        "baseline_test_macro_f1": baseline_macro_f1,
        "fixed_test_macro_f1": row["fixed_test_macro_f1"],
        "tuned_val_macro_f1": row["val_macro_f1_mean"],
        "tuned_test_macro_f1": row["test_macro_f1_mean"],
        "tuned_test_balanced_accuracy": row["test_balanced_accuracy_mean"],
        "tuned_delta_macro_f1": row["test_macro_f1_mean"] - baseline_macro_f1,
    }


def _with_baseline_rows(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for benchmark in ("patch_feature", "wsi_bag"):
        part = frame[frame["benchmark"] == benchmark]
        if part.empty:
            continue
        baseline_macro_f1 = float(part.iloc[0]["baseline_test_macro_f1"])
        rows.append(
            {
                "benchmark": benchmark,
                "regime": REGIME_LABEL[benchmark],
                "method": BASELINE_METHOD[benchmark],
                "method_label": _method_label(BASELINE_METHOD[benchmark]),
                "selected_params": "fixed baseline",
                "baseline_method": BASELINE_METHOD[benchmark],
                "baseline_test_macro_f1": baseline_macro_f1,
                "fixed_test_macro_f1": baseline_macro_f1,
                "tuned_val_macro_f1": pd.NA,
                "tuned_test_macro_f1": baseline_macro_f1,
                "tuned_test_balanced_accuracy": pd.NA,
                "tuned_delta_macro_f1": 0.0,
            }
        )
        rows.extend(dict(row) for _, row in part.iterrows())
    return pd.DataFrame(rows)


def _fixed_summary(paths: dict[str, Path], benchmark: str) -> pd.DataFrame:
    stem = SUMMARY_STEM[benchmark]
    return pd.read_csv(paths["tables"] / f"result_summary_{stem}.csv")


def _fixed_metric(frame: pd.DataFrame, method: str, metric: str) -> float:
    lookup = PATCH_FEATURE_METHOD_ALIASES.get(method, method)
    candidates = [method, lookup]
    matches = frame[(frame["split"] == "test") & (frame["method"].isin(candidates))]
    if matches.empty:
        raise KeyError(
            f"No fixed-protocol test row for method {method!r} (lookup={lookup!r})"
        )
    return float(matches.iloc[0][f"{metric}_mean"])


def _read_result(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _prefixed_metrics(prefix: str, result: dict[str, Any]) -> dict[str, float]:
    return {f"{prefix}_{metric}": float(result[metric]) for metric in METRICS}


def _method_label(method: str) -> str:
    return METHOD_LABELS.get(method, method.replace("_", " "))


if __name__ == "__main__":
    main()
