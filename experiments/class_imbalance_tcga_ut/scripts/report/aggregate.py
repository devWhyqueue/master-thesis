from __future__ import annotations

import logging
import argparse
import gzip
import json
from pathlib import Path
from typing import Any, cast

import pandas as pd

from scripts.common import ensure_dirs, load_config, write_json
from scripts.mil.metadata import method_metadata

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for result aggregation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    return parser.parse_args()


def read_result(path: Path, split: str) -> dict | None:
    """Read a split result JSON if present."""
    result_path = path / f"{split}_results.json"
    if split == "val" and not result_path.exists():
        result_path = path / "validation_results.json"
    if not result_path.exists():
        return None
    with result_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _result_row(
    method: str, seed: int, split: str, result: dict[str, Any]
) -> dict[str, Any]:
    """Build one row for seed-level summary."""
    metadata = method_metadata(method)
    return {
        "method": method,
        "role": metadata["role"],
        "taxonomy_category": metadata["taxonomy_category"],
        "representative_paper": metadata["representative_paper"],
        "fidelity": metadata["fidelity"],
        "seed": seed,
        "split": split,
        "accuracy": result["accuracy"],
        "balanced_accuracy": result["balanced_accuracy"],
        "macro_precision": result["macro_precision"],
        "macro_recall": result["macro_recall"],
        "macro_f1": result["macro_f1"],
    }


def _detail_row(
    method: str, seed: int, split: str, result: dict[str, Any]
) -> dict[str, Any]:
    """Build one row preserving detailed per-class results."""
    return {
        "method": method,
        "method_metadata": method_metadata(method),
        "seed": seed,
        "split": split,
        "result": _compact_result(result),
    }


def _compact_result(result: dict[str, Any]) -> dict[str, Any]:
    """Keep metrics needed for analysis without raw prediction arrays."""
    raw_array_keys = {"labels", "preds", "probabilities"}
    return {key: value for key, value in result.items() if key not in raw_array_keys}


def _collect_rows(
    config: dict[str, Any], results_root: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Collect available result rows and missing entries."""
    rows: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for method in config["methods"]:
        for seed in config["training"]["seeds"]:
            result_dir = results_root / method / f"seed={seed}"
            for split in ["val", "test"]:
                result = read_result(result_dir, split)
                if result is None:
                    missing.append({"method": method, "seed": seed, "split": split})
                    continue
                rows.append(_result_row(method, seed, split, result))
                details.append(_detail_row(method, seed, split, result))
    return rows, details, missing


def _aggregate_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Aggregate seed-level rows to method-level statistics."""
    if summary.empty:
        return pd.DataFrame()
    grouped = summary.groupby(["method", "split"], as_index=False).agg(
        role=("role", "first"),
        taxonomy_category=("taxonomy_category", "first"),
        representative_paper=("representative_paper", "first"),
        fidelity=("fidelity", "first"),
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
        balanced_accuracy_std=("balanced_accuracy", "std"),
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),
    )
    rows = [
        {
            "method": str(row["method"]),
            "role": str(row["role"]),
            "taxonomy_category": str(row["taxonomy_category"]),
            "representative_paper": str(row["representative_paper"]),
            "fidelity": str(row["fidelity"]),
            "split": str(row["split"]),
            "accuracy_mean": float(row["accuracy_mean"]),
            "accuracy_std": float(row["accuracy_std"]),
            "balanced_accuracy_mean": float(row["balanced_accuracy_mean"]),
            "balanced_accuracy_std": float(row["balanced_accuracy_std"]),
            "macro_f1_mean": float(row["macro_f1_mean"]),
            "macro_f1_std": float(row["macro_f1_std"]),
        }
        for _, row in grouped.iterrows()
    ]
    ordered_rows = sorted(
        rows,
        key=lambda row: (row["split"], -row["macro_f1_mean"]),
    )
    return pd.DataFrame(ordered_rows)


def _write_summary_tables(
    summary: pd.DataFrame, aggregate: pd.DataFrame, tables_dir: Path
) -> None:
    """Write CSV and LaTeX table outputs."""
    summary_path = tables_dir / "result_summary_by_seed.csv"
    aggregate_path = tables_dir / "result_summary.csv"
    tex_path = tables_dir / "result_summary.tex"
    summary.to_csv(summary_path, index=False)
    aggregate.to_csv(aggregate_path, index=False)
    write_latex_table(aggregate, tex_path)
    logger.info(f"Wrote {summary_path}")
    logger.info(f"Wrote {aggregate_path}")
    logger.info(f"Wrote {tex_path}")


def _write_detail_archive(rows: list[dict[str, Any]], path: Path) -> None:
    """Write detailed per-run results to one compressed JSONL artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")
    logger.info(f"Wrote {path}")


def main() -> None:
    """Aggregate experiment metrics across methods and seeds."""
    args = parse_args()
    config = load_config(args.config)
    paths = ensure_dirs(config)
    rows, details, missing = _collect_rows(config, paths["results"])
    summary = pd.DataFrame(rows)
    aggregate = _aggregate_summary(summary)
    _write_summary_tables(summary, aggregate, paths["tables"])
    _write_detail_archive(details, paths["tables"] / "result_details.jsonl.gz")
    write_json(paths["tables"] / "missing_results.json", {"missing": missing})


def write_latex_table(frame: pd.DataFrame, path: Path) -> None:
    """Render result table as LaTeX."""
    if frame.empty:
        path.write_text(
            "\\begin{tabular}{lccc}\\toprule Method & Balanced accuracy & Macro F1 & Split\\\\\\midrule "
            "Pending & -- & -- & --\\\\\\bottomrule\\end{tabular}\n",
            encoding="utf-8",
        )
        return
    test_frame = _latex_rows_source(frame)
    lines = _latex_header_lines()
    lines.extend(_latex_data_lines(test_frame))
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _latex_rows_source(frame: pd.DataFrame) -> pd.DataFrame:
    """Choose and sort rows for LaTeX output."""
    test_frame = frame[frame["split"] == "test"]
    selected = test_frame if not test_frame.empty else frame
    rows = [
        {
            "method": str(row["method"]),
            "split": str(row["split"]),
            "accuracy_mean": float(row["accuracy_mean"]),
            "accuracy_std": float(row["accuracy_std"]),
            "balanced_accuracy_mean": float(row["balanced_accuracy_mean"]),
            "balanced_accuracy_std": float(row["balanced_accuracy_std"]),
            "macro_f1_mean": float(row["macro_f1_mean"]),
            "macro_f1_std": float(row["macro_f1_std"]),
        }
        for _, row in selected.iterrows()
    ]
    ordered_rows = sorted(rows, key=lambda row: -row["macro_f1_mean"])
    return pd.DataFrame(ordered_rows)


def _latex_header_lines() -> list[str]:
    """Build static LaTeX table header."""
    return [
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Method & Balanced accuracy & Macro F1 & Split\\\\",
        "\\midrule",
    ]


def _latex_data_lines(frame: pd.DataFrame) -> list[str]:
    """Build LaTeX rows from aggregate data."""
    lines: list[str] = []
    for _, row in cast(pd.DataFrame, frame).iterrows():
        method = str(row["method"]).replace("_", "\\_")
        balanced = f"{float(row['balanced_accuracy_mean']):.3f}"
        macro_f1 = f"{float(row['macro_f1_mean']):.3f}"
        lines.append(f"{method} & {balanced} & {macro_f1} & {row['split']}\\\\")
    return lines


if __name__ == "__main__":
    main()
