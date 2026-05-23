from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from scripts.common import ensure_dirs, load_config
from scripts.report.figures_metrics import (
    benchmark_title,
    plot_macro_f1_by_seed,
    plot_macro_f1_delta,
)
from scripts.training.support_tiers import load_class_tier_labels

METHOD_LABELS = {
    "patch_ce": "CE",
    "patch_weighted_ce": "Weighted CE",
    "patch_focal": "Focal",
    "patch_balanced_sampler_ce": "Balanced sampler",
    "patch_ce_soft_f1_balanced": "CE + soft F1 (balanced)",
    "patch_ce_soft_mcc_balanced": "CE + soft MCC (balanced)",
    "patch_progan_aug": "ProGAN augmentation",
    "patch_feature_ce": "CE",
    "patch_feature_weighted_ce": "Weighted CE",
    "patch_feature_focal": "Focal",
    "patch_feature_balanced_sampler_ce": "Balanced sampler",
    "patch_feature_ce_soft_f1_balanced": "CE + soft F1 (balanced)",
    "patch_feature_ce_soft_mcc_balanced": "CE + soft MCC (balanced)",
    "patch_feature_progan_aug": "ProGAN augmentation",
    "mil_ce": "MIL CE",
    "mil_weighted_ce": "Weighted MIL",
    "mil_focal": "Focal MIL",
    "mil_balanced_sampler_ce": "Balanced MIL",
    "rankmix_mil": "RankMix",
    "sc_mil": "SC-MIL",
    "mde_mil": "MDE-MIL (ensemble)",
}


def parse_args() -> argparse.Namespace:
    """Parse figure-generation arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--benchmark", required=True, choices=["patch", "wsi_bag"])
    parser.add_argument("--split", default="test", choices=["val", "test"])
    return parser.parse_args()


def _methods(config: dict, benchmark: str) -> list[str]:
    return list(
        config["patch_feature_methods"]
        if benchmark == "patch"
        else config["wsi_bag_methods"]
    )


def _method_label(method: str) -> str:
    return METHOD_LABELS.get(method, method.replace("_", " "))


def _classwise_rows(method: str, seed: int, result: dict) -> list[dict[str, object]]:
    return [
        {
            "method": method,
            "seed": seed,
            "class_name": class_name,
            "recall": recall,
            "support": support,
        }
        for class_name, recall, support in zip(
            result["class_names"],
            result["recall_per_class"],
            result["support_per_class"],
            strict=False,
        )
    ]


def _load_archive(path: Path, methods: list[str], split: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            if payload["method"] in methods and payload["split"] == split:
                rows.extend(
                    _classwise_rows(
                        payload["method"], int(payload["seed"]), payload["result"]
                    )
                )
    return pd.DataFrame(rows)


def _tier_pivot(
    frame: pd.DataFrame, methods: list[str], tier_labels: dict[str, str]
) -> pd.DataFrame:
    """Aggregate classwise recalls into predefined support tiers."""
    grouped = cast(
        pd.DataFrame,
        frame.groupby(["method", "class_name"], as_index=False)["recall"].mean(),
    )
    grouped["tier"] = grouped["class_name"].map(
        lambda name: tier_labels[str(name)].capitalize()
    )
    return grouped.pivot_table(
        index="method", columns="tier", values="recall", aggfunc="mean"
    ).reindex(methods)


def _plot_tier_heatmap(pivot: pd.DataFrame, path: Path, benchmark: str) -> None:
    """Render one support-tier heatmap."""
    columns = [column for column in ["Tail", "Body", "Head"] if column in pivot.columns]
    values = pivot[columns].to_numpy()
    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    image = ax.imshow(values, cmap="plasma", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(
        np.arange(len(columns)), labels=[f"{column} recall" for column in columns]
    )
    ax.set_yticks(
        np.arange(len(pivot.index)),
        labels=[_method_label(method) for method in pivot.index],
    )
    ax.set_title(f"{benchmark_title(benchmark)} mean recall by support tier")
    _write_values(ax, values)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04).set_label("Mean recall")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_classwise_recall(
    archive: Path,
    methods: list[str],
    path: Path,
    split: str,
    benchmark: str,
    tables_path: Path,
) -> None:
    """Plot mean recall by support tier for one benchmark."""
    if not archive.exists():
        return
    frame = _load_archive(archive, methods, split)
    class_names = sorted(frame["class_name"].astype(str).unique().tolist())
    tier_labels = load_class_tier_labels(
        class_names, tables_path / "class_distribution.csv"
    )
    if tier_labels is None:
        return
    _plot_tier_heatmap(_tier_pivot(frame, methods, tier_labels), path, benchmark)


def _write_values(ax: Axes, values: np.ndarray) -> None:
    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            value = values[row_idx, col_idx]
            ax.text(
                col_idx,
                row_idx,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if value < 0.72 else "black",
            )


def main() -> None:
    """Generate benchmark-specific publication figures."""
    args = parse_args()
    config = load_config(args.config)
    paths = ensure_dirs(config)
    methods = _methods(config, args.benchmark)
    archive = paths["tables"] / f"result_details_{args.benchmark}.jsonl.gz"
    if args.benchmark == "patch":
        plot_macro_f1_delta(
            archive,
            methods,
            paths["figures"]
            / f"method_macro_f1_delta_{args.benchmark}_{args.split}.png",
            args.split,
            args.benchmark,
            METHOD_LABELS,
        )
    else:
        plot_macro_f1_by_seed(
            archive,
            methods,
            paths["figures"]
            / f"method_macro_f1_by_seed_{args.benchmark}_{args.split}.png",
            args.split,
            args.benchmark,
            METHOD_LABELS,
        )
    plot_classwise_recall(
        archive,
        methods,
        paths["figures"] / f"classwise_recall_{args.benchmark}_{args.split}.png",
        args.split,
        args.benchmark,
        paths["tables"],
    )


if __name__ == "__main__":
    main()
