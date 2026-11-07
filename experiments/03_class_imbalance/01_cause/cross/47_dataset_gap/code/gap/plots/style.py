"""Shared figure style: fixed dataset and role colours, axis chrome, and file output.

TCGA-UT is blue and BRACS orange in every figure (categorical slots 1-2). Scatter plots highlight
a confusion pair in aqua and violet, distinct from the dataset hues, and grey the other classes.
Gained and lost predictions use the blue/red diverging pair.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

__all__ = [
    "DATASETS",
    "LABEL",
    "COLOR",
    "PAIRS",
    "HIGHLIGHT",
    "LOST",
    "GAINED",
    "MUTED",
    "INK",
    "save",
    "load_analyses",
    "bare_axis",
]

DATASETS = ("tcga_ut", "bracs")
LABEL = {"tcga_ut": "TCGA-UT", "bracs": "BRACS"}
COLOR = {"tcga_ut": "#2a78d6", "bracs": "#eb6834"}
PAIRS = {
    "tcga_ut": ("Colon_adenocarcinoma", "Rectum_adenocarcinoma"),
    "bracs": ("ADH", "DCIS"),
}
HIGHLIGHT = ("#1baf7a", "#6250d6")
LOST, GAINED, MUTED, INK = "#e34948", "#2a78d6", "#b9b8b2", "#52514e"

plt.rcParams.update(
    {
        "font.size": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": INK,
        "axes.labelcolor": "#0b0b0b",
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.grid": True,
        "grid.color": "#e6e5e0",
        "grid.linewidth": 0.5,
        "legend.frameon": False,
        "savefig.bbox": "tight",
    }
)


def save(fig: Figure, path: Path, dpi: int | None = None) -> None:
    """Write and close one figure."""
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def load_analyses(report_dir: Path) -> dict[str, dict[str, Any]]:
    """Both datasets' exported analysis summaries."""
    return {
        ds: json.loads((report_dir / f"analysis_{ds}.json").read_text(encoding="utf-8"))
        for ds in DATASETS
    }


def bare_axis(ax: Axes) -> None:
    """Remove ticks, grid, and spines, for image panels."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
