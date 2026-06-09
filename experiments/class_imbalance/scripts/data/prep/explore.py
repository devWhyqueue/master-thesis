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

from scripts.common import ensure_dirs, load_config, write_json

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
    slide_manifest: pd.DataFrame, figures_dir: Path, tables_dir: Path
) -> dict[str, Any]:
    """Write class distribution tables and plots, then return summary stats."""
    counts = slide_manifest["cancer_type"].value_counts().sort_values()
    counts_df = counts.to_frame(name="n_slides").rename_axis("cancer_type")
    counts_df.reset_index().assign(rank_ascending=range(1, len(counts_df) + 1)).to_csv(
        tables_dir / "class_distribution.csv", index=False
    )
    _write_distribution_outputs(counts, figures_dir)
    return _distribution_stats(counts)


def _write_distribution_outputs(counts: pd.Series, figures_dir: Path) -> None:
    _plot_class_distribution(counts, figures_dir)
    _plot_long_tail_profile(counts, figures_dir)


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


def _plot_long_tail_profile(counts: pd.Series, figures_dir: Path) -> None:
    tier_frame = _support_tier_frame(counts)
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    bars = ax.bar(
        tier_frame["tier"],
        tier_frame["slides"],
        color=[
            TIER_COLORS["tail"],
            TIER_COLORS["body"],
            "#6aa56f",
            TIER_COLORS["head"],
        ],
    )
    ax.set_ylabel("Slides")
    ax.set_xlabel("Class-support tier")
    ax.set_title("Slide mass by support tier")
    labels = [
        f"{int(slides)} slides\n{int(classes)} classes"
        for slides, classes in zip(
            tier_frame["slides"], tier_frame["classes"], strict=False
        )
    ]
    ax.bar_label(bars, labels=labels, padding=3, fontsize=8)
    ax.margins(y=0.18)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "long_tail_profile.png", dpi=300)
    plt.close(fig)


def _support_tier_frame(counts: pd.Series) -> pd.DataFrame:
    tier = pd.cut(
        counts,
        bins=[0, 100, 250, 500, float("inf")],
        labels=["<100", "100-249", "250-499", ">=500"],
        right=False,
    )
    frame = pd.DataFrame({"slides": counts, "tier": tier})
    grouped = frame.groupby("tier", observed=False).agg(
        slides=("slides", "sum"),
        classes=("slides", "size"),
    )
    return cast(pd.DataFrame, grouped.reset_index())


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
        split_counts.size().to_frame("n_slides").reset_index().to_csv(
            paths["tables"] / f"split_distribution_seed={seed}.csv", index=False
        )
        stats["split_slide_counts"] = split_manifest["split"].value_counts().to_dict()
        return stats
    cached_path = paths["tables"] / f"split_distribution_seed={seed}.csv"
    if cached_path.exists():
        split_distribution = pd.read_csv(cached_path)
        stats["split_slide_counts"] = (
            split_distribution.groupby("split")["n_slides"].sum().to_dict()
        )
    return stats


def _read_counts(paths: dict[str, Path]) -> pd.Series:
    counts_path = paths["tables"] / "class_distribution.csv"
    if not counts_path.exists():
        raise FileNotFoundError(
            "Neither data/slide_manifest.csv nor outputs/tables/class_distribution.csv "
            "is available."
        )
    counts = cast(
        pd.Series, pd.read_csv(counts_path).set_index("cancer_type")["n_slides"]
    )
    return cast(pd.Series, counts.sort_values())


def main() -> None:
    """Generate dataset-level statistics and diagnostic plots."""
    args = parse_args()
    config = load_config(args.config)
    paths = ensure_dirs(config)
    slide_manifest_path = paths["data"] / "slide_manifest.csv"
    if slide_manifest_path.exists():
        slide_manifest = pd.read_csv(slide_manifest_path)
        stats = save_class_distribution(
            slide_manifest, paths["figures"], paths["tables"]
        )
    else:
        counts = _read_counts(paths)
        _write_distribution_outputs(counts, paths["figures"])
        stats = _distribution_stats(counts)
    stats = _add_split_statistics(stats, paths, args.seed)
    write_json(paths["tables"] / "dataset_stats.json", stats)


if __name__ == "__main__":
    main()
