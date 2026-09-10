"""Plotting utilities for the patient-breadth experiment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from breadth import BREADTH_LADDER, DEPTH_LADDER

__all__ = ["generate_figures"]


def _plot_breadth_curve(ax: Any, g: int, cell_accs: dict[str, dict[str, Any]]) -> None:
    """Plot accuracy vs nominal patch budget for one breadth level."""
    m_vals = np.array(DEPTH_LADDER)
    n_vals = g * m_vals
    points = [cell_accs[f"G{g}_m{m}"]["point"] for m in m_vals]
    err_low = [
        max(0.0, points[i] - cell_accs[f"G{g}_m{m}"]["ci_2_5"])
        for i, m in enumerate(m_vals)
    ]
    err_high = [
        max(0.0, cell_accs[f"G{g}_m{m}"]["ci_97_5"] - points[i])
        for i, m in enumerate(m_vals)
    ]
    ax.errorbar(
        n_vals,
        points,
        yerr=[err_low, err_high],
        fmt="o-",
        label=f"G={g} patients",
        capsize=3,
    )


def generate_figures(results: dict[str, Any], dest_dir: Path) -> None:
    """Generate plots visualizing the support surface and contrasts."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    cell_accs = results["cell_accuracies"]
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    for g in BREADTH_LADDER:
        _plot_breadth_curve(ax, g, cell_accs)

    iso_cells = [(20, 8), (10, 16), (5, 32)]
    iso_pts = [cell_accs[f"G{g}_m{m}"]["point"] for g, m in iso_cells]
    ax.scatter([160, 160, 160], iso_pts, color="red", zorder=5, s=60, marker="x")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Nominal class patch budget $n_c = G \\times m$")
    ax.set_ylabel("Patient-macro balanced accuracy (%)")
    ax.set_title("Breadth x Depth Support Surface")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(dest_dir / "support_surface_nominal.png")
    plt.close(fig)

