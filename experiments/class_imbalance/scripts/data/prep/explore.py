from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast

from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
from matplotlib.ticker import ScalarFormatter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import json

from scripts.common import ensure_dirs, load_config
from scripts.analysis.results import connect, init_schema, replace_table

ABBREVIATIONS = {
    "Brain_Lower_Grade_Glioma": "LGG",
    "Breast_invasive_carcinoma": "BRCA",
    "Cholangiocarcinoma": "CHOL",
    "Glioblastoma_multiforme": "GBM",
    "Lymphoid_Neoplasm_Diffuse_Large_B-cell_Lymphoma": "DLBC",
    "Pheochromocytoma_and_Paraganglioma": "PCPG",
}
TIER_COLORS = {
    "head": "#c8615a",
    "body": "#d8a657",
    "tail": "#6b9ac4",
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for dataset exploration."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def save_class_distribution(
    slide_manifest: pd.DataFrame, figures_dir: Path
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Write class distribution tables and plots, then return summary stats."""
    counts = slide_manifest["cancer_type"].value_counts().sort_values()
    counts_df = counts.to_frame(name="n_slides").rename_axis("cancer_type")
    distribution = counts_df.reset_index().assign(
        rank_ascending=range(1, len(counts_df) + 1)
    )
    _write_distribution_outputs(counts, figures_dir)
    return distribution, _distribution_stats(counts)


def _write_distribution_outputs(counts: pd.Series, figures_dir: Path) -> None:
    _plot_class_distribution(counts, figures_dir)


def _plot_class_distribution(counts: pd.Series, figures_dir: Path) -> None:
    descending = counts.sort_values(ascending=False)
    ranks = np.arange(1, len(descending) + 1)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(
        ranks,
        np.asarray(descending.to_numpy()),
        color=_rank_colors(len(descending)),
        width=0.84,
    )
    ax.set_yscale("log")
    ax.set_ylim(20, 1000)
    ax.set_yticks([25, 50, 100, 200, 400, 800])
    ax.get_yaxis().set_major_formatter(ScalarFormatter())
    ax.set_ylabel("Slides per class (log scale)")
    ax.set_xlabel("Cancer type rank by slide count")
    ax.set_title("Native TCGA-UT long-tail class distribution")
    ax.grid(axis="y", which="major", alpha=0.25)
    ax.legend(handles=_rank_legend_handles(), frameon=False, loc="upper right")
    _annotate_extremes(ax, descending)
    fig.tight_layout()
    fig.savefig(figures_dir / "class_distribution.png", dpi=300)
    plt.close(fig)


def _rank_colors(n_classes: int) -> list[str]:
    head_cut = 8
    tail_start = n_classes - 8
    return [
        TIER_COLORS["head"]
        if rank <= head_cut
        else TIER_COLORS["body"]
        if rank <= tail_start
        else TIER_COLORS["tail"]
        for rank in range(1, n_classes + 1)
    ]


def _rank_legend_handles() -> list[Rectangle]:
    return [
        Rectangle((0, 0), 1, 1, color=TIER_COLORS["head"], label="Head classes"),
        Rectangle((0, 0), 1, 1, color=TIER_COLORS["body"], label="Body classes"),
        Rectangle((0, 0), 1, 1, color=TIER_COLORS["tail"], label="Tail classes"),
    ]


def _annotate_extremes(ax: Axes, descending: pd.Series) -> None:
    labels = [str(label) for label in descending.index]
    counts = descending.to_numpy()
    head_lines = [_count_label(labels, counts, idx) for idx in range(3)]
    tail_range = range(len(descending) - 3, len(descending))
    tail_lines = [_count_label(labels, counts, idx) for idx in tail_range]
    box = {"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85}
    ax.text(
        0.03,
        0.95,
        "Largest classes\n" + "\n".join(head_lines),
        transform=ax.transAxes,
        va="top",
        fontsize=8,
        bbox=box,
    )
    ax.text(
        0.62,
        0.18,
        "Smallest classes\n" + "\n".join(tail_lines),
        transform=ax.transAxes,
        va="bottom",
        fontsize=8,
        bbox=box,
    )


def _count_label(labels: list[str], counts: np.ndarray, idx: int) -> str:
    label = ABBREVIATIONS.get(labels[idx], labels[idx].replace("_", " ")[:32])
    return f"{label}: {int(counts[idx])}"


def _distribution_stats(counts: pd.Series) -> dict[str, Any]:
    return {
        "n_classes": int(len(counts)),
        "n_slides": int(counts.sum()),
        "min_slides_per_class": int(counts.min()),
        "max_slides_per_class": int(counts.max()),
        "imbalance_ratio": float(counts.max() / counts.min()),
    }


def _add_split_statistics(
    stats: dict[str, Any], paths: dict[str, Path], seed: int
) -> dict[str, Any]:
    split_path = paths["data"] / f"slide_splits_seed={seed}.csv"
    if split_path.exists():
        split_manifest = pd.read_csv(split_path)
        split_counts = split_manifest.groupby(["split", "cancer_type"])
        split_distribution = split_counts.size().to_frame("n_slides").reset_index()
        split_distribution["seed"] = seed
        stats["split_distribution"] = split_distribution
        stats["split_slide_counts"] = split_manifest["split"].value_counts().to_dict()
        return stats
    return stats


def _read_counts(paths: dict[str, Path]) -> pd.Series:
    connection = connect(paths["db"])
    init_schema(connection)
    distribution = pd.read_sql('SELECT * FROM "dataset_class_distribution"', connection)
    connection.close()
    if not distribution.empty:
        counts = cast(pd.Series, distribution.set_index("cancer_type")["n_slides"])
        return cast(pd.Series, counts.sort_values())
    counts_path = paths["tables"] / "class_distribution.csv"
    if counts_path.exists():
        counts = cast(
            pd.Series, pd.read_csv(counts_path).set_index("cancer_type")["n_slides"]
        )
        return cast(pd.Series, counts.sort_values())
    raise FileNotFoundError(
        "Neither data/slide_manifest.csv nor dataset class distribution is available."
    )


def main() -> None:
    """Generate dataset-level statistics and diagnostic plots."""
    args = parse_args()
    config = load_config(args.config)
    paths = ensure_dirs(config)
    slide_manifest_path = paths["data"] / "slide_manifest.csv"
    if slide_manifest_path.exists():
        slide_manifest = pd.read_csv(slide_manifest_path)
        distribution, stats = save_class_distribution(slide_manifest, paths["figures"])
    else:
        counts = _read_counts(paths)
        _write_distribution_outputs(counts, paths["figures"])
        distribution = counts.to_frame(name="n_slides").rename_axis("cancer_type")
        distribution = distribution.reset_index().assign(
            rank_ascending=range(1, len(distribution) + 1)
        )
        stats = _distribution_stats(counts)
    stats = _add_split_statistics(stats, paths, args.seed)
    _publish_dataset_tables(paths, distribution, stats)


def _publish_dataset_tables(
    paths: dict[str, Path], distribution: pd.DataFrame, stats: dict[str, Any]
) -> None:
    connection = connect(paths["db"])
    init_schema(connection)
    replace_table(connection, "dataset_class_distribution", distribution)
    split_distribution = stats.pop("split_distribution", None)
    if isinstance(split_distribution, pd.DataFrame):
        replace_table(connection, "dataset_split_distribution", split_distribution)
    replace_table(
        connection,
        "dataset_stats",
        pd.DataFrame([{"payload_json": json.dumps(stats, sort_keys=True)}]),
    )
    connection.close()


if __name__ == "__main__":
    main()
