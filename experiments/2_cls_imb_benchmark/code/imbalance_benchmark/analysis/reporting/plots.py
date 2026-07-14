from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from imbalance_benchmark.analysis.calibration import reliability_curve
from imbalance_benchmark.analysis.metrics import assign_tiers
from imbalance_benchmark.analysis.query import load_seed_predictions

__all__ = [
    "plot_confusion_matrix",
    "plot_reliability_diagram",
    "plot_tail_vs_support",
    "write_tail_reliability",
]

# Headless rendering: figures are only ever saved to disk, never shown interactively.
plt.switch_backend("Agg")


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], path: Path) -> None:
    """Row-normalized confusion matrix heatmap."""
    cm = np.asarray(cm, dtype=np.float64)
    row_sums = cm.sum(axis=1, keepdims=True)
    normalized = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)
    fig, ax = plt.subplots(
        figsize=(max(4, len(class_names) * 0.6), max(4, len(class_names) * 0.6))
    )
    im = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_reliability_diagram(
    centers: np.ndarray, mean_confidence: np.ndarray, accuracy: np.ndarray, path: Path
) -> None:
    """Reliability diagram: mean confidence vs. empirical accuracy per confidence bin."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect calibration")
    ax.bar(
        centers, accuracy, width=0.08, alpha=0.7, edgecolor="black", label="accuracy"
    )
    ax.scatter(mean_confidence, accuracy, color="black", zorder=3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_tail_vs_support(classwise: pd.DataFrame, path: Path) -> None:
    """Tail-tier recall against allocated support, one point per (condition, method, class)."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for tier, group in classwise.groupby("tier"):
        ax.scatter(group["support"], group["recall"], label=str(tier), alpha=0.7)
    ax.set_xlabel("Support (allocated training count)")
    ax.set_ylabel("Recall")
    ax.legend(title="Tier")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_tail_reliability(
    paths: dict[str, Path], freeze: dict[str, Any], balanced: dict[str, Any]
) -> None:
    """Write reliability diagrams for tail classes."""
    for assignment, conditions in freeze.get("assignment_conditions", {}).items():
        allocated = conditions.get("severe", {}).get("allocated_counts", {})
        names = balanced["class_names"]
        order = freeze.get("tail_assignments", {}).get(assignment, names)
        tiers = assign_tiers(names, allocated, order)
        tail = [i for i, n in enumerate(names) if tiers.get(n) == "tail"]
        src = load_seed_predictions(paths, "severe", "ce", assignment) or balanced
        mask = np.isin(src["labels"], tail)
        if mask.any():
            p, y = src["probs"].mean(axis=0)[mask], src["labels"][mask]
            centers, conf, acc = reliability_curve(p, y)
            if len(centers):
                fp = paths["figures"] / f"tail_reliability_{assignment}.png"
                plot_reliability_diagram(centers, conf, acc, fp)
