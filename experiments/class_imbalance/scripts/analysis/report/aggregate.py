from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import pandas as pd

from scripts.common import ensure_dirs, load_config
from scripts.analysis.results import (
    SUMMARY_METRICS,
    connect,
    init_schema,
    load_eval_details,
    read_table,
    replace_table,
)
from scripts.metadata import benchmark_metadata
from scripts.analysis.report.figures.labels import latex_method_label


def parse_args() -> argparse.Namespace:
    """Parse result-aggregation arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--benchmark", required=True, choices=["patch", "wsi_bag"])
    return parser.parse_args()


def _settings(
    config: dict[str, Any], benchmark: str
) -> tuple[list[str], list[int], str]:
    if benchmark == "patch":
        return (
            list(config["patch_feature_methods"]),
            list(config["patch_feature_training"]["seeds"]),
            "patch_feature",
        )
    return (
        list(config["wsi_bag_methods"]),
        list(config["wsi_training"]["seeds"]),
        "wsi_bag",
    )


def _collect(
    config: dict[str, Any], paths: dict[str, Path], benchmark: str
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    methods, seeds, _ = _settings(config, benchmark)
    connection = connect(paths["db"])
    init_schema(connection)
    rows: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for split in ("val", "test"):
        split_details = load_eval_details(connection, benchmark, methods, seeds, split)
        details.extend(split_details)
        for payload in split_details:
            result = payload["result"]
            metadata = payload.get("method_metadata") or benchmark_metadata(
                benchmark, payload["method"]
            )
            rows.append(
                {
                    "benchmark": benchmark,
                    "method": payload["method"],
                    **metadata,
                    "seed": payload["seed"],
                    "split": split,
                    **{key: result[key] for key in SUMMARY_METRICS if key in result},
                }
            )
    for method in methods:
        for seed in seeds:
            for split in ("val", "test"):
                if not any(
                    item["method"] == method
                    and item["seed"] == seed
                    and item["split"] == split
                    for item in details
                ):
                    missing.append({"method": method, "seed": seed, "split": split})
    connection.close()
    return pd.DataFrame(rows), details, missing


def _aggregate(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    grouped = summary.groupby(["benchmark", "method", "split"], as_index=False).agg(
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
        label = latex_method_label(method_key)
        lines.append(
            f"{label} & {_format_mean_std(row, 'accuracy')} & "
            f"{_format_mean_std(row, 'balanced_accuracy')} & "
            f"{_format_mean_std(row, 'macro_f1')}\\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _upsert_benchmark_table(
    connection,
    table_name: str,
    frame: pd.DataFrame,
    benchmark: str,
) -> None:
    existing = read_table(connection, table_name)
    if not existing.empty and "benchmark" in existing.columns:
        existing = existing[existing["benchmark"] != benchmark]
        frame = pd.concat([existing, frame], ignore_index=True)
    replace_table(connection, table_name, frame)


def _upsert_missing_results(
    connection, missing: list[dict[str, Any]], benchmark: str
) -> None:
    payload = {"missing": missing, "benchmark": benchmark}
    existing = read_table(connection, "missing_results")
    rows = []
    if not existing.empty:
        for raw in existing["payload_json"]:
            rows.append(json.loads(raw))
    rows = [row for row in rows if row.get("benchmark") != benchmark]
    rows.append(payload)
    replace_table(
        connection,
        "missing_results",
        pd.DataFrame(
            [{"payload_json": json.dumps(row, sort_keys=True)} for row in rows]
        ),
    )


def _format_mean_std(row: dict[str, Any], metric: str) -> str:
    mean = float(row[f"{metric}_mean"])
    std = float(row[f"{metric}_std"])
    return rf"$\num{{{mean:.3f}}} \pm \num{{{std:.3f}}}$"


def main() -> None:
    """Aggregate one benchmark independently from the other."""
    args = parse_args()
    config = load_config(args.config)
    paths = ensure_dirs(config)
    summary, _details, missing = _collect(config, paths, args.benchmark)
    aggregate = _aggregate(summary)
    connection = connect(paths["db"])
    init_schema(connection)
    _upsert_benchmark_table(connection, "summary_by_seed", summary, args.benchmark)
    _upsert_benchmark_table(connection, "summary", aggregate, args.benchmark)
    _upsert_missing_results(connection, missing, args.benchmark)
    connection.close()
    stem = f"result_summary_{args.benchmark}"
    _write_latex(aggregate, paths["tables"] / f"{stem}.tex")


if __name__ == "__main__":
    main()
