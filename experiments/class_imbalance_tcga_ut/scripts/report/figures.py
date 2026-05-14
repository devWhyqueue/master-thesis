from __future__ import annotations

import argparse
import gzip
import json
import logging
from pathlib import Path
from typing import cast

from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.common import ensure_dirs, load_config

logger = logging.getLogger(__name__)
METHOD_LABELS = {
    "ce": "CE",
    "weighted_ce": "Weighted CE",
    "focal": "Focal",
    "balanced_sampler_ce": "Balanced sampler",
    "knn": "KNN",
    "ncc": "Nearest centroid",
    "rankmix_mil": "RankMix MIL",
    "feature_gan_mil": "Feature synth. MIL",
    "cfal_mil": "CFAL MIL",
    "mde_mil": "MDE-MIL",
    "sc_mil": "SC-MIL",
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for figure generation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    return parser.parse_args()


def plot_metric_summary(summary: pd.DataFrame, figures_dir: Path, split: str) -> None:
    """Plot method-level macro-F1 summary for a split."""
    filtered = cast(pd.DataFrame, summary.loc[summary["split"] == split, :])
    split_summary = cast(pd.DataFrame, filtered.sort_values("macro_f1_mean"))
    if split_summary.empty:
        return
    labels = [_method_label(method) for method in split_summary["method"]]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(
        labels,
        split_summary["macro_f1_mean"],
        xerr=split_summary["macro_f1_std"],
        color="#4f7cac",
        alpha=0.9,
        capsize=3,
    )
    ax.set_xlabel("Macro F1")
    ax.set_title(f"Method comparison on {split} split")
    ax.set_xlim(0.62, 0.90)
    ax.grid(axis="x", alpha=0.25)
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
    """Plot support-tier recall aggregated over seeds."""
    frame = _load_classwise_rows(results_dir, methods, seeds, split)
    if frame.empty:
        archive = results_dir.parent / "tables" / "result_details.jsonl.gz"
        frame = _load_classwise_archive(archive, methods, split)
    if frame.empty:
        return
    ordered_classes = _ordered_classes_by_support(frame)
    grouped = cast(
        pd.DataFrame,
        frame.groupby(["method", "class_name"], as_index=False)["recall"].mean(),
    )
    _plot_support_tier_heatmap(grouped, methods, ordered_classes, figures_dir, split)


def _load_classwise_rows(
    results_dir: Path, methods: list[str], seeds: list[int], split: str
) -> pd.DataFrame:
    rows = []
    for method in methods:
        for seed in seeds:
            result = _read_split_result(results_dir, method, seed, split)
            if result is None:
                continue
            rows.extend(_classwise_rows(method, seed, result))
    return pd.DataFrame(rows)


def _read_split_result(
    results_dir: Path, method: str, seed: int, split: str
) -> dict | None:
    result_path = results_dir / method / f"seed={seed}" / f"{split}_results.json"
    if not result_path.exists():
        return None
    with result_path.open("r", encoding="utf-8") as handle:
        return cast(dict, json.load(handle))


def _load_classwise_archive(
    archive_path: Path, methods: list[str], split: str
) -> pd.DataFrame:
    if not archive_path.exists():
        return pd.DataFrame()
    rows = []
    method_set = set(methods)
    with gzip.open(archive_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            method = str(payload["method"])
            if method in method_set and payload["split"] == split:
                rows.extend(
                    _classwise_rows(method, int(payload["seed"]), payload["result"])
                )
    return pd.DataFrame(rows)


def _classwise_rows(method: str, seed: int, result: dict) -> list[dict]:
    recalls = zip(
        result["class_names"],
        result["recall_per_class"],
        result["support_per_class"],
        strict=False,
    )
    return [
        {
            "method": method,
            "seed": seed,
            "class_name": class_name,
            "recall": recall,
            "support": support,
        }
        for class_name, recall, support in recalls
    ]


def _ordered_classes_by_support(frame: pd.DataFrame) -> list[str]:
    support = cast(pd.Series, frame.groupby("class_name")["support"].mean())
    pairs = sorted(
        ((str(name), float(value)) for name, value in support.to_dict().items()),
        key=lambda pair: pair[1],
    )
    return [name for name, _ in pairs]


def _plot_support_tier_heatmap(
    grouped: pd.DataFrame,
    methods: list[str],
    ordered_classes: list[str],
    figures_dir: Path,
    split: str,
) -> None:
    tier_frame = _support_tier_recall(grouped, methods, ordered_classes)
    if tier_frame.empty:
        return
    columns = [
        column
        for column in ["Tail recall", "Body recall", "Head recall"]
        if column in tier_frame.columns
    ]
    if not columns:
        return
    values = tier_frame[columns].to_numpy()
    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    image = ax.imshow(values, cmap="YlGnBu", vmin=0.55, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(columns)), labels=columns)
    ax.set_yticks(
        np.arange(len(tier_frame.index)),
        labels=[_method_label(method) for method in tier_frame.index],
    )
    ax.set_title(f"Mean recall by class-support tier ({split})")
    _write_heatmap_values(ax, values)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04).set_label("Mean recall")
    fig.tight_layout()
    fig.savefig(figures_dir / f"classwise_recall_{split}.png", dpi=300)
    plt.close(fig)


def _support_tier_recall(
    grouped: pd.DataFrame, methods: list[str], ordered_classes: list[str]
) -> pd.DataFrame:
    tail = set(ordered_classes[:8])
    head = set(ordered_classes[-8:])
    tier_map = {
        class_name: "Tail recall"
        if class_name in tail
        else "Head recall"
        if class_name in head
        else "Body recall"
        for class_name in ordered_classes
    }
    frame = grouped.copy()
    frame["tier"] = frame["class_name"].map(lambda name: tier_map[str(name)])
    pivot = frame.pivot_table(
        index="method", columns="tier", values="recall", aggfunc="mean"
    )
    ordered_methods = [method for method in methods if method in pivot.index]
    return cast(pd.DataFrame, pivot.reindex(ordered_methods))


def _write_heatmap_values(ax: Axes, values: np.ndarray) -> None:
    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            value = values[row_idx, col_idx]
            ax.text(
                col_idx,
                row_idx,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value < 0.70 else "#1f2933",
                fontsize=8,
            )


def _method_label(method: str) -> str:
    return METHOD_LABELS.get(str(method), str(method).replace("_", " "))


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
    logger.info("Wrote figures under %s", paths["figures"])


if __name__ == "__main__":
    main()
