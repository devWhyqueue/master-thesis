from __future__ import annotations

import logging
import argparse
import json
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import pandas as pd

from scripts.common import ensure_dirs, load_config

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for figure generation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    return parser.parse_args()


def plot_metric_summary(summary: pd.DataFrame, figures_dir: Path, split: str) -> None:
    """Plot method-level macro-F1 summary for a split."""
    filtered = cast(pd.DataFrame, summary.loc[summary["split"] == split, :])
    split_summary = cast(
        pd.DataFrame,
        filtered.sort_values(by=["macro_f1_mean"]),
    )
    if split_summary.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(split_summary["method"], split_summary["macro_f1_mean"])
    ax.set_xlabel("Macro F1")
    ax.set_title(f"Method comparison ({split})")
    fig.tight_layout()
    fig.savefig(figures_dir / f"method_macro_f1_{split}.png", dpi=300)
    plt.close(fig)


def plot_classwise_recall(
    results_dir: Path,
    methods: list[str],
    seeds: list[int],
    figures_dir: Path,
    split: str,
) -> None:
    """Plot classwise recall curves aggregated over seeds."""
    frame = _load_classwise_rows(results_dir, methods, seeds, split)
    if frame.empty:
        return
    ordered_classes = _ordered_classes_by_support(frame)
    grouped = cast(
        pd.DataFrame,
        frame.groupby(["method", "class_name"], as_index=False)["recall"].mean(),
    )
    _plot_classwise_frame(grouped, methods, ordered_classes, figures_dir, split)


def _load_classwise_rows(
    results_dir: Path, methods: list[str], seeds: list[int], split: str
) -> pd.DataFrame:
    """Load classwise recall rows from result JSON files."""
    rows = []
    for method in methods:
        for seed in seeds:
            result_path = (
                results_dir / method / f"seed={seed}" / f"{split}_results.json"
            )
            if not result_path.exists():
                continue
            with result_path.open("r", encoding="utf-8") as handle:
                result = json.load(handle)
            supports = result["support_per_class"]
            for class_name, recall, support in zip(
                result["class_names"],
                result["recall_per_class"],
                supports,
                strict=False,
            ):
                rows.append(
                    {
                        "method": method,
                        "seed": seed,
                        "class_name": class_name,
                        "recall": recall,
                        "support": support,
                    }
                )
    return pd.DataFrame(rows)


def _ordered_classes_by_support(frame: pd.DataFrame) -> list[str]:
    """Order classes by average support."""
    grouped = cast(pd.Series, frame.groupby("class_name")["support"].mean())
    support_map = grouped.to_dict()
    ordered_pairs = sorted(
        ((str(name), float(value)) for name, value in support_map.items()),
        key=lambda pair: pair[1],
    )
    return [name for name, _ in ordered_pairs]


def _plot_classwise_frame(
    grouped: pd.DataFrame,
    methods: list[str],
    ordered_classes: list[str],
    figures_dir: Path,
    split: str,
) -> None:
    """Draw and save classwise recall plot."""
    fig, ax = plt.subplots(figsize=(11, 5))
    for method in methods:
        method_frame = grouped[grouped["method"] == method]
        if method_frame.empty:
            continue
        values = method_frame.set_index("class_name").reindex(ordered_classes)["recall"]
        ax.plot(ordered_classes, values, marker="o", label=method)
    ax.set_ylabel("Recall")
    ax.set_xlabel("Cancer type ordered by support")
    ax.set_title(f"Classwise recall by method ({split})")
    ax.tick_params(axis="x", rotation=75, labelsize=7)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(figures_dir / f"classwise_recall_{split}.png", dpi=300)
    plt.close(fig)


def main() -> None:
    """Generate publication figures from aggregate outputs."""
    args = parse_args()
    config = load_config(args.config)
    paths = ensure_dirs(config)
    summary_path = paths["tables"] / "result_summary.csv"
    if summary_path.exists():
        plot_metric_summary(pd.read_csv(summary_path), paths["figures"], args.split)
    plot_classwise_recall(
        paths["results"],
        config["methods"],
        list(config["training"]["seeds"]),
        paths["figures"],
        args.split,
    )
    logger.info(f"Wrote figures under {paths['figures']}")


if __name__ == "__main__":
    main()
