from __future__ import annotations

import gzip
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASELINE_METHOD = {"patch": "patch_ce", "wsi_bag": "mil_ce"}
BASELINE_LABEL = {"patch": "CE", "wsi_bag": "MIL CE"}


def benchmark_title(benchmark: str) -> str:
    """Return a publication-ready benchmark label."""
    return "Patch benchmark" if benchmark == "patch" else "WSI-bag benchmark"


def _load_macro_f1_table(
    archive: Path, methods: list[str], split: str
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    with gzip.open(archive, "rt", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            if payload["method"] in methods and payload["split"] == split:
                rows.append(
                    {
                        "method": payload["method"],
                        "seed": int(payload["seed"]),
                        "macro_f1": float(payload["result"]["macro_f1"]),
                    }
                )
    return pd.DataFrame(rows)


def plot_macro_f1_delta(
    archive: Path,
    methods: list[str],
    path: Path,
    split: str,
    benchmark: str,
    method_label: dict[str, str],
) -> None:
    """Plot per-seed macro-F1 change relative to the regime baseline."""
    frame = _load_macro_f1_table(archive, methods, split)
    baseline_key = BASELINE_METHOD[benchmark]
    baseline = frame.loc[frame["method"] == baseline_key].set_index("seed")[
        "macro_f1"
    ]
    rows: list[dict[str, object]] = []
    for method in methods:
        if method == baseline_key:
            continue
        part = frame.loc[frame["method"] == method].set_index("seed")
        diff = part["macro_f1"] - baseline
        rows.append(
            {
                "method": method,
                "delta_mean": float(diff.mean()),
                "delta_std": float(diff.std(ddof=0)),
            }
        )
    if not rows:
        return
    table = pd.DataFrame(rows).sort_values("delta_mean")
    colors = ["#c44e52" if value < 0 else "#4c9a6a" for value in table["delta_mean"]]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(
        [method_label.get(method, method) for method in table["method"]],
        table["delta_mean"],
        xerr=table["delta_std"],
        color=colors,
        capsize=3,
    )
    ax.axvline(0.0, color="#333333", linewidth=0.9)
    ax.set_xlabel(
        rf"$\Delta$ macro F1 vs. {BASELINE_LABEL[benchmark]} baseline"
    )
    ax.set_title(f"{benchmark_title(benchmark)} ({split})")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_macro_f1_by_seed(
    archive: Path,
    methods: list[str],
    path: Path,
    split: str,
    benchmark: str,
    method_label: dict[str, str],
) -> None:
    """Plot macro F1 across seeds for each method in one benchmark."""
    frame = _load_macro_f1_table(archive, methods, split)
    if frame.empty:
        return
    baseline_key = BASELINE_METHOD[benchmark]
    seeds = sorted(frame["seed"].unique())
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for method in methods:
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
            continue
        ax.plot(part["seed"], part["macro_f1"], label=label, **style)
    ax.set_xticks(seeds, labels=[str(seed) for seed in seeds])
    ax.set_xlabel("Seed")
    ax.set_ylabel("Macro F1")
    ax.set_title(f"{benchmark_title(benchmark)} ({split})")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
