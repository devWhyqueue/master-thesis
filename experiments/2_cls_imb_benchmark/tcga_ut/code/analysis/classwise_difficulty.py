from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable, cast

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import pandas as pd

from code.common import ensure_dirs, load_config
from code.analysis.results import (
    connect,
    init_schema,
    load_class_distribution,
    load_split_payload,
    replace_table,
)

BASELINES = {"patch": "patch_feature_ce", "wsi_bag": "mil_ce"}
BENCHMARK_LABELS = {"patch": "Patch", "wsi_bag": "WSI-bag"}
LABEL_OFFSETS = {"Cholangiocarcinoma": (5, 12), "Uterine_Carcinosarcoma": (8, -1)}


def parse_args() -> argparse.Namespace:
    """Parse classwise difficulty analysis arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    """Compute classwise difficulty artifacts for the CE baselines."""
    args = parse_args()
    config = load_config(args.config)
    paths = ensure_dirs(config)
    frame = _difficulty_frame(paths, config, args.split)
    correlations = _correlation_frame(frame)
    connection = connect(paths["db"])
    init_schema(connection)
    stored = frame.copy()
    stored["split"] = args.split
    replace_table(connection, "classwise_difficulty", stored)
    replace_table(connection, "classwise_difficulty_correlations", correlations)
    connection.close()
    _write_hardest_table(
        _hardest_classes(frame, args.top_k),
        paths["tables"] / f"classwise_difficulty_hardest_{args.split}.tex",
    )
    for benchmark in BASELINES:
        _plot_benchmark(
            cast(pd.DataFrame, frame.loc[frame["benchmark"] == benchmark]),
            paths["figures"] / f"classwise_difficulty_{benchmark}_{args.split}.png",
            benchmark,
        )


def _difficulty_frame(paths: dict[str, Path], config: dict, split: str) -> pd.DataFrame:
    connection = connect(paths["db"])
    init_schema(connection)
    class_support = _read_class_support(load_class_distribution(connection, paths))
    rows = []
    for benchmark, method in BASELINES.items():
        seeds = (
            list(config["patch_feature_training"]["seeds"])
            if benchmark == "patch"
            else list(config["wsi_training"]["seeds"])
        )
        for seed in seeds:
            result = load_split_payload(connection, benchmark, method, seed, split)
            if result is None:
                continue
            payload = {"method": method, "seed": seed, "result": result}
            rows.extend(_result_rows(payload, benchmark, class_support))
    connection.close()
    grouped = _aggregate_classes(pd.DataFrame(rows))
    return _add_ranks(grouped)


def _read_class_support(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = frame.rename(
        columns={"cancer_type": "class_name", "n_slides": "dataset_support"}
    )[["class_name", "dataset_support", "rank_ascending"]]
    return cast(pd.DataFrame, renamed)


def _result_rows(
    payload: dict[str, Any], benchmark: str, support: pd.DataFrame
) -> list[dict[str, Any]]:
    result = payload["result"]
    metric_rows = zip(
        result["class_names"],
        result["precision_per_class"],
        result["recall_per_class"],
        result["f1_per_class"],
        result["support_per_class"],
        strict=False,
    )
    indexed = support.set_index("class_name")
    return [
        {
            "benchmark": benchmark,
            "method": payload["method"],
            "seed": int(payload["seed"]),
            "class_name": class_name,
            "dataset_support": int(indexed.loc[class_name, "dataset_support"]),
            "support_rank_ascending": int(indexed.loc[class_name, "rank_ascending"]),
            "test_support": int(test_support),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "error_rate": 1.0 - float(recall),
        }
        for class_name, precision, recall, f1, test_support in metric_rows
    ]


def _aggregate_classes(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = cast(
        pd.DataFrame,
        frame.groupby(["benchmark", "method", "class_name"], as_index=False).agg(
            dataset_support=("dataset_support", "first"),
            support_rank_ascending=("support_rank_ascending", "first"),
            test_support_mean=("test_support", "mean"),
            precision_mean=("precision", "mean"),
            recall_mean=("recall", "mean"),
            f1_mean=("f1", "mean"),
            error_rate_mean=("error_rate", "mean"),
        ),
    )
    return grouped.sort_values(by=["benchmark", "support_rank_ascending"])


def _add_ranks(frame: pd.DataFrame) -> pd.DataFrame:
    ranked = frame.copy()
    ranked["difficulty_rank"] = ranked.groupby("benchmark")["error_rate_mean"].rank(
        method="min", ascending=False
    )
    return ranked.sort_values(by=["benchmark", "difficulty_rank", "class_name"])


def _correlation_frame(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for benchmark, group in frame.groupby("benchmark"):
        rows.append(
            {
                "benchmark": benchmark,
                "method": BASELINES[str(benchmark)],
                "n_classes": len(group),
                "support_vs_recall_spearman": _spearman(
                    group["dataset_support"], group["recall_mean"]
                ),
                "support_vs_error_spearman": _spearman(
                    group["dataset_support"], group["error_rate_mean"]
                ),
                "support_rank_vs_difficulty_rank_spearman": _spearman(
                    group["support_rank_ascending"], group["difficulty_rank"]
                ),
            }
        )
    return pd.DataFrame(rows)


def _spearman(left: Iterable[float], right: Iterable[float]) -> float:
    left_rank = pd.Series(list(left), dtype="float64").rank()
    right_rank = pd.Series(list(right), dtype="float64").rank()
    return float(left_rank.corr(right_rank))


def _hardest_classes(frame: pd.DataFrame, top_k: int) -> pd.DataFrame:
    hardest = frame.sort_values(by=["benchmark", "difficulty_rank"]).groupby(
        "benchmark", as_index=False
    )
    return hardest.head(top_k).sort_values(by=["benchmark", "difficulty_rank"])


def _write_hardest_table(frame: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{tabularx}{\linewidth}{@{}lXrrrrr@{}}",
        r"\toprule",
        r"Regime & Class & Slides & Support rank & Difficulty rank & Recall & F1\\",
        r"\midrule",
        *[_table_row(row) for row in frame.to_dict("records")],
        r"\bottomrule",
        r"\end{tabularx}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _table_row(row: dict[str, Any]) -> str:
    label = BENCHMARK_LABELS[str(row["benchmark"])]
    class_name = str(row["class_name"]).replace("_", " ")
    support = int(row["dataset_support"])
    support_rank = int(row["support_rank_ascending"])
    difficulty_rank = int(row["difficulty_rank"])
    recall = float(row["recall_mean"])
    f1 = float(row["f1_mean"])
    return (
        f"{label} & {class_name} & \\num{{{support}}} & {support_rank} & "
        f"{difficulty_rank} & \\num{{{recall:.3f}}} & \\num{{{f1:.3f}}}\\\\"
    )


def _plot_benchmark(frame: pd.DataFrame, path: Path, benchmark: str) -> None:
    plot_frame = frame.sort_values("support_rank_ascending")
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    ax.scatter(
        plot_frame["support_rank_ascending"],
        plot_frame["error_rate_mean"],
        s=42,
        color="#2f6f8f",
        edgecolor="white",
        linewidth=0.7,
    )
    _label_hard_cases(ax, plot_frame)
    ax.set_xlabel("Class support rank (1 = lowest slide support)")
    ax.set_ylabel("CE baseline error rate (1 - recall)")
    ax.set_title(f"{BENCHMARK_LABELS[benchmark]} class difficulty vs. support")
    ax.grid(True, axis="y", color="#d9d9d9", linewidth=0.8)
    ax.set_xlim(0.2, 32.8)
    ax.set_ylim(0.0, max(0.55, float(plot_frame["error_rate_mean"].max()) + 0.08))
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _label_hard_cases(ax: Axes, frame: pd.DataFrame) -> None:
    label_frame = frame.nsmallest(5, "recall_mean")
    for offset, row in enumerate(label_frame.to_dict("records")):
        class_name = str(row["class_name"])
        ax.annotate(
            _short_label(class_name),
            (row["support_rank_ascending"], row["error_rate_mean"]),
            xytext=LABEL_OFFSETS.get(class_name, (5, 6 + 3 * (offset % 2))),
            textcoords="offset points",
            fontsize=7.5,
        )


def _short_label(name: str) -> str:
    return " ".join(name.replace("_", " ").split()[:3])


if __name__ == "__main__":
    main()
