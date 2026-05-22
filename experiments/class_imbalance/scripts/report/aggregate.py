from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any, cast

import pandas as pd

from scripts.common import ensure_dirs, load_config, write_json
from scripts.metadata import benchmark_metadata
from scripts.report.figures import METHOD_LABELS

SUMMARY_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "negative_log_likelihood",
    "brier_score",
    "expected_calibration_error",
)


def parse_args() -> argparse.Namespace:
    """Parse result-aggregation arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--benchmark", required=True, choices=["patch", "wsi_bag"])
    return parser.parse_args()


def _settings(
    config: dict[str, Any], benchmark: str
) -> tuple[list[str], list[int], str, str | None]:
    if benchmark == "patch":
        return (
            list(config["patch_feature_methods"]),
            list(config["patch_feature_training"]["seeds"]),
            "results",
            "patch_feature",
        )
    return (
        list(config["wsi_bag_methods"]),
        list(config["wsi_training"]["seeds"]),
        "wsi_results",
        None,
    )


def _read(path: Path, split: str) -> dict[str, Any] | None:
    result_path = path / f"{split}_results.json"
    if not result_path.exists():
        return None
    with result_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _collect(
    config: dict[str, Any], paths: dict[str, Path], benchmark: str
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    methods, seeds, result_key, result_subdir = _settings(config, benchmark)
    for method in methods:
        for seed in seeds:
            result_base = paths[result_key]
            if result_subdir is not None:
                result_base = result_base / result_subdir
            result_dir = result_base / method / f"seed={seed}"
            for split in ["val", "test"]:
                result = _read(result_dir, split)
                if result is None:
                    missing.append({"method": method, "seed": seed, "split": split})
                    continue
                metadata = benchmark_metadata(benchmark, method)
                rows.append(
                    {
                        "method": method,
                        **metadata,
                        "seed": seed,
                        "split": split,
                        **{
                            key: result[key] for key in SUMMARY_METRICS if key in result
                        },
                    }
                )
                details.append(
                    {
                        "method": method,
                        "method_metadata": metadata,
                        "seed": seed,
                        "split": split,
                        "result": {
                            key: value
                            for key, value in result.items()
                            if key not in {"labels", "preds", "probabilities"}
                        },
                    }
                )
    if not rows:
        details = _read_existing_details(paths, benchmark)
        rows = _summary_rows_from_details(details)
        if rows:
            missing = []
    return pd.DataFrame(rows), details, missing


def _read_existing_details(
    paths: dict[str, Path], benchmark: str
) -> list[dict[str, Any]]:
    path = paths["tables"] / f"result_details_{benchmark}.jsonl.gz"
    if not path.exists():
        return []
    details = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            details.append(json.loads(line))
    return details


def _summary_rows_from_details(details: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for payload in details:
        result = payload["result"]
        rows.append(
            {
                "method": payload["method"],
                **payload["method_metadata"],
                "seed": payload["seed"],
                "split": payload["split"],
                **{key: result[key] for key in SUMMARY_METRICS if key in result},
            }
        )
    return rows


def _aggregate(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    grouped = summary.groupby(["method", "split"], as_index=False).agg(
        role=("role", "first"),
        taxonomy_category=("taxonomy_category", "first"),
        representative_paper=("representative_paper", "first"),
        **{
            f"{metric}_mean": (metric, "mean")
            for metric in SUMMARY_METRICS
            if metric in summary.columns
        },
        **{
            f"{metric}_std": (metric, "std")
            for metric in SUMMARY_METRICS
            if metric in summary.columns
        },
    )
    rows = sorted(
        grouped.iterrows(),
        key=lambda item: (str(item[1]["split"]), -float(item[1]["macro_f1_mean"])),
    )
    return pd.DataFrame([row for _, row in rows])


def _write_latex(frame: pd.DataFrame, path: Path) -> None:
    selected = cast(pd.DataFrame, frame[frame["split"] == "test"])
    test_rows = sorted(
        selected.iterrows(), key=lambda item: -float(item[1]["macro_f1_mean"])
    )
    test = pd.DataFrame([row for _, row in test_rows])
    lines = [
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Method & Accuracy & Balanced accuracy & Macro F1\\\\",
        "\\midrule",
    ]
    for row in test.to_dict("records"):
        method_key = str(row["method"])
        label = METHOD_LABELS.get(method_key, method_key.replace("_", " "))
        lines.append(
            f"{label} & {_format_mean_std(row, 'accuracy')} & "
            f"{_format_mean_std(row, 'balanced_accuracy')} & "
            f"{_format_mean_std(row, 'macro_f1')}\\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _format_mean_std(row: dict[str, Any], metric: str) -> str:
    mean = float(row[f"{metric}_mean"])
    std = float(row[f"{metric}_std"])
    return rf"$\num{{{mean:.3f}}} \pm \num{{{std:.3f}}}$"


def main() -> None:
    """Aggregate one benchmark independently from the other."""
    args = parse_args()
    config = load_config(args.config)
    paths = ensure_dirs(config)
    summary, details, missing = _collect(config, paths, args.benchmark)
    aggregate = _aggregate(summary)
    stem = f"result_summary_{args.benchmark}"
    summary.to_csv(paths["tables"] / f"{stem}_by_seed.csv", index=False)
    aggregate.to_csv(paths["tables"] / f"{stem}.csv", index=False)
    _write_latex(aggregate, paths["tables"] / f"{stem}.tex")
    with gzip.open(
        paths["tables"] / f"result_details_{args.benchmark}.jsonl.gz",
        "wt",
        encoding="utf-8",
    ) as handle:
        for row in details:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    write_json(
        paths["tables"] / f"missing_results_{args.benchmark}.json", {"missing": missing}
    )


if __name__ == "__main__":
    main()
