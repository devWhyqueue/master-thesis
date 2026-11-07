"""Balanced-decision figures: patient-macro confusion and the true-class margin distribution."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from gap.plots.codes import short_name
from gap.plots.style import COLOR, DATASETS, INK, LABEL, save

__all__ = ["confusion_figure", "margin_figure"]

_MARGIN_LABEL = (
    r"margin: $\log p_{\mathrm{true}} - \max_{\mathrm{rival}} \log p$ (nats)"
)


def _cell_labels(ax: Any, m: np.ndarray) -> None:
    """Percentages printed in every cell, white on dark cells."""
    for (i, j), v in np.ndenumerate(m):
        color = "white" if v > 50 else "#0b0b0b"
        ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6.5, color=color)


def confusion_figure(analysis: dict[str, Any], path: Path, annotate: bool) -> None:
    """Patient-macro r1 confusion (rows = true class, each summing to 100 %)."""
    m = np.asarray(analysis["confusion_r1"]) * 100.0
    names = [short_name(n) for n in analysis["class_names"]]
    fig, ax = plt.subplots(figsize=(3.2, 2.9) if annotate else (6.3, 6.0))
    ax.imshow(m, cmap="Blues", vmin=0, vmax=100)
    ax.grid(False)
    font = 7 if annotate else 6
    ax.set_xticks(
        range(len(names)), names, rotation=0 if annotate else 90, fontsize=font
    )
    ax.set_yticks(range(len(names)), names, fontsize=font)
    ax.set_xlabel("predicted class")
    ax.set_ylabel("true class")
    if annotate:
        _cell_labels(ax, m)
    save(fig, path)


def _margin_curves(ax: Any, margin: dict[str, Any], ds: str) -> None:
    """Raw and temperature-scaled CDFs plus the error-rate point at zero margin."""
    grid, color = np.asarray(margin["grid"]), COLOR[ds]
    ax.plot(grid, margin["cdf_raw"], color=color, lw=1.8, label=f"{LABEL[ds]}, raw")
    scaled = f"{LABEL[ds]}, temperature-scaled"
    ax.plot(grid, margin["cdf_scaled"], color=color, lw=1.2, ls="--", label=scaled)
    error = 1.0 - margin["positive_mass"]
    ax.plot([0], [error], "o", color=color, markersize=5, markeredgecolor="white")
    offset = (6, -10 if ds == "tcga_ut" else 4)
    ax.annotate(
        f"{error * 100:.1f}% wrong",
        (0, error),
        xytext=offset,
        textcoords="offset points",
        fontsize=7,
        color=INK,
    )


def margin_figure(data: dict[str, dict[str, Any]], path: Path) -> None:
    """Patient-macro weighted CDF of the true-class margin, raw and temperature-scaled."""
    fig, ax = plt.subplots(figsize=(4.2, 2.7))
    for ds in DATASETS:
        _margin_curves(ax, data[ds]["margin"], ds)
    ax.axvline(0, color=INK, linewidth=0.8)
    ax.set_xlim(-8, 8)
    ax.set_ylim(0, 1)
    ax.set_xlabel(_MARGIN_LABEL)
    ax.set_ylabel("patient-macro cumulative share")
    ax.legend(fontsize=6.5, loc="upper left")
    save(fig, path)
