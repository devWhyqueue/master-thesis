from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from scripts.common import ensure_dirs, load_config
from scripts.analysis.results import (
    connect,
    init_schema,
    load_class_distribution,
    load_eval_details,
)
from scripts.analysis.report.figures.labels import METHOD_LABELS, method_label
from scripts.analysis.report.figures.metrics import (
    benchmark_title,
    plot_macro_f1_by_seed,
)
from scripts.modeling.training.support_tiers import class_tier_labels


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


def _seeds(config: dict, benchmark: str) -> list[int]:
    if benchmark == "patch":
        return list(config["patch_feature_training"]["seeds"])
    return list(config["wsi_training"]["seeds"])


def _method_label(method: str) -> str:
    return method_label(method)


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


def _details_frame(
    paths: dict[str, Path],
    config: dict,
    benchmark: str,
    split: str,
) -> pd.DataFrame:
    connection = connect(paths["db"])
    init_schema(connection)
    details = load_eval_details(
        connection,
        benchmark,
        _methods(config, benchmark),
        _seeds(config, benchmark),
        split,
    )
    connection.close()
    rows: list[dict[str, object]] = []
    for payload in details:
        rows.extend(
            _classwise_rows(payload["method"], int(payload["seed"]), payload["result"])
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
    paths: dict[str, Path],
    config: dict,
    methods: list[str],
    path: Path,
    split: str,
    benchmark: str,
) -> None:
    """Plot mean recall by support tier for one benchmark."""
    frame = _details_frame(paths, config, benchmark, split)
    if frame.empty:
        return
    class_names = sorted(frame["class_name"].astype(str).unique().tolist())
    connection = connect(paths["db"])
    init_schema(connection)
    distribution = load_class_distribution(connection, paths)
    connection.close()
    if distribution.empty:
        return
    slide_counts = dict(
        zip(
            distribution["cancer_type"],
            distribution["n_slides"].astype(int),
            strict=True,
        )
    )
    tier_labels = class_tier_labels(class_names, slide_counts)
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


def _benchmark_details(
    paths: dict[str, Path], config: dict, benchmark: str, split: str
) -> list[dict[str, object]]:
    connection = connect(paths["db"])
    init_schema(connection)
    details = load_eval_details(
        connection,
        benchmark,
        _methods(config, benchmark),
        _seeds(config, benchmark),
        split,
    )
    connection.close()
    return details


def main() -> None:
    """Generate benchmark-specific publication figures."""
    args = parse_args()
    config = load_config(args.config)
    paths = ensure_dirs(config)
    methods = _methods(config, args.benchmark)
    details = _benchmark_details(paths, config, args.benchmark, args.split)
    if args.benchmark == "wsi_bag":
        plot_macro_f1_by_seed(
            details,
            methods,
            paths["figures"]
            / f"method_macro_f1_by_seed_{args.benchmark}_{args.split}.png",
            args.split,
            args.benchmark,
            METHOD_LABELS,
        )
    plot_classwise_recall(
        paths,
        config,
        methods,
        paths["figures"] / f"classwise_recall_{args.benchmark}_{args.split}.png",
        args.split,
        args.benchmark,
    )


if __name__ == "__main__":
    main()
