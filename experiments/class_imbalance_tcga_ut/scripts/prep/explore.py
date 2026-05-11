from __future__ import annotations

import logging
import argparse
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.common import ensure_dirs, load_config, write_json

logger = logging.getLogger(__name__)


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
    _write_class_distribution_table(counts, tables_dir)
    _plot_class_distribution(counts, figures_dir)
    _plot_long_tail_profile(counts, figures_dir)
    return _distribution_stats(counts)


def _write_class_distribution_table(counts: pd.Series, tables_dir: Path) -> None:
    """Write per-class slide counts table."""
    counts_df = (
        counts.to_frame(name="n_slides").rename_axis("cancer_type").reset_index()
    )
    counts_df["rank_ascending"] = range(1, len(counts_df) + 1)
    counts_df.to_csv(tables_dir / "class_distribution.csv", index=False)


def _plot_class_distribution(counts: pd.Series, figures_dir: Path) -> None:
    """Plot class distribution bar chart."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(list(map(str, counts.index.tolist())), np.asarray(counts.to_numpy()))
    ax.set_ylabel("Slides")
    ax.set_xlabel("Cancer type")
    ax.set_title("Native TCGA-UT class distribution")
    ax.tick_params(axis="x", rotation=75, labelsize=8)
    fig.tight_layout()
    fig.savefig(figures_dir / "class_distribution.png", dpi=300)
    plt.close(fig)


def _plot_long_tail_profile(counts: pd.Series, figures_dir: Path) -> None:
    """Plot long-tail profile by class rank."""
    fig, ax = plt.subplots(figsize=(6, 4))
    descending = counts.sort_values(ascending=False).to_numpy()
    ax.plot(range(1, len(descending) + 1), descending, marker="o")
    ax.set_xlabel("Class rank")
    ax.set_ylabel("Slides")
    ax.set_title("Long-tail profile")
    fig.tight_layout()
    fig.savefig(figures_dir / "long_tail_profile.png", dpi=300)
    plt.close(fig)


def _distribution_stats(counts: pd.Series) -> dict[str, Any]:
    """Compute scalar class-distribution statistics."""
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
    """Attach optional split-level counts when split manifest exists."""
    split_path = paths["data"] / f"slide_splits_seed={seed}.csv"
    if not split_path.exists():
        return stats
    split_manifest = pd.read_csv(split_path)
    split_counts = cast(
        pd.DataFrame,
        split_manifest.groupby(["split", "cancer_type"])
        .size()
        .to_frame("n_slides")
        .reset_index(),
    )
    split_counts.to_csv(
        paths["tables"] / f"split_distribution_seed={seed}.csv", index=False
    )
    stats["split_slide_counts"] = split_manifest["split"].value_counts().to_dict()
    return stats


def main() -> None:
    """Generate dataset-level statistics and diagnostic plots."""
    args = parse_args()
    config = load_config(args.config)
    paths = ensure_dirs(config)

    slide_manifest = pd.read_csv(paths["data"] / "slide_manifest.csv")
    stats = save_class_distribution(slide_manifest, paths["figures"], paths["tables"])
    stats = _add_split_statistics(stats, paths, args.seed)
    write_json(paths["tables"] / "dataset_stats.json", stats)
    logger.info(
        "Wrote dataset exploration outputs under %s and %s",
        paths["figures"],
        paths["tables"],
    )


if __name__ == "__main__":
    main()
