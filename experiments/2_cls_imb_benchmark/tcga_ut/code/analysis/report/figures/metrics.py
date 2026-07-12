from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import pandas as pd

BASELINE_METHOD = {"patch": "patch_feature_ce", "wsi_bag": "mil_ce"}


def benchmark_title(benchmark: str) -> str:
    """Return a publication-ready benchmark label."""
    return "Patch benchmark" if benchmark == "patch" else "WSI-bag benchmark"


def macro_f1_frame(details: list[dict[str, Any]], methods: list[str]) -> pd.DataFrame:
    """Build a per-seed macro-F1 table from result detail payloads."""
    rows: list[dict[str, object]] = []
    for payload in details:
        if payload["method"] not in methods:
            continue
        rows.append(
            {
                "method": payload["method"],
                "seed": int(payload["seed"]),
                "macro_f1": float(payload["result"]["macro_f1"]),
            }
        )
    return pd.DataFrame(rows)


def _plot_method_line(
    ax: Axes,
    frame: pd.DataFrame,
    method: str,
    baseline_key: str,
    method_label: dict[str, str],
) -> None:
    part = frame.loc[frame["method"] == method].sort_values("seed")
    label = method_label.get(method, method)
    style = {"linewidth": 2.2, "marker": "o", "markersize": 5}
    if method == baseline_key:
        ax.plot(
            part["seed"],
            part["macro_f1"],
            label=f"{label} (baseline)",
            color="#333333",
            linestyle="--",
            **style,
        )
        return
    ax.plot(part["seed"], part["macro_f1"], label=label, **style)


def plot_macro_f1_by_seed(
    details: list[dict[str, Any]],
    methods: list[str],
    path: Path,
    split: str,
    benchmark: str,
    method_label: dict[str, str],
) -> None:
    """Plot macro F1 across seeds for each method in one benchmark."""
    frame = macro_f1_frame(details, methods)
    if frame.empty:
        return
    baseline_key = BASELINE_METHOD[benchmark]
    seeds = sorted(frame["seed"].unique())
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for method in methods:
        _plot_method_line(ax, frame, method, baseline_key, method_label)
    ax.set_xticks(seeds, labels=[str(seed) for seed in seeds])
    ax.set_xlabel("Seed")
    ax.set_ylabel("Macro F1")
    ax.set_title(f"{benchmark_title(benchmark)} ({split})")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
