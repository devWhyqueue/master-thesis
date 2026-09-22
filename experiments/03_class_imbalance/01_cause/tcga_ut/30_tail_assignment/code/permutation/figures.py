"""Figures: the sampled-permutation spread at rho 100, and per-class predicted tail loss vs r1 recall."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from prevalence.figures import _save

__all__ = ["spread_figure", "tail_loss_figure"]

_MARKER_COLOR = {"N": "tab:red", "easy_r100": "tab:green", "hard_r100": "tab:orange"}


def _plot_markers(ax: Axes, markers: dict[str, np.ndarray]) -> None:
    """One point + 95% CI whisker per named reference distribution, at y = 0."""
    for name, dist in markers.items():
        point = float(dist[0])
        lo, hi = np.percentile(dist[1:], [2.5, 97.5])
        ax.errorbar(
            [point],
            [0.0],
            xerr=[[point - lo], [hi - point]],
            fmt="D",
            color=_MARKER_COLOR.get(name, "black"),
            label=name,
            zorder=5,
        )


def spread_figure(
    samples: np.ndarray,
    native_draws: np.ndarray,
    markers: dict[str, np.ndarray],
    dest: Path,
) -> None:
    """Histogram of the sampled-permutation predicted D (rho 100), with observed reference markers."""
    fig, ax = plt.subplots(figsize=(6, 3.8), dpi=200)
    ax.hist(
        samples,
        bins=60,
        color="tab:blue",
        alpha=0.5,
        density=True,
        label="sampled permutations (model)",
    )
    for i, x in enumerate(native_draws):
        ax.axvline(
            x,
            color="tab:gray",
            lw=0.6,
            alpha=0.5,
            label="exp-25 draws (observed)" if i == 0 else None,
        )
    _plot_markers(ax, markers)
    ax.set_xlabel("BA damage vs. r1 (pp)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=7)
    _save(fig, dest)


def tail_loss_figure(
    own_loss_by_rho: dict[int, np.ndarray], r1_recall: list[float], dest: Path
) -> None:
    """Per-class predicted tail own loss vs. r1 recall, both ratios."""
    fig, ax = plt.subplots(figsize=(5, 4), dpi=200)
    colors = {10: "tab:blue", 100: "tab:red"}
    for rho, own_loss in own_loss_by_rho.items():
        ax.scatter(
            r1_recall,
            own_loss[:, 0],
            s=14,
            color=colors.get(rho, "black"),
            label=rf"$\rho$ = {rho}",
        )
    ax.axhline(0.0, color="black", lw=0.8, alpha=0.5)
    ax.set_xlabel("r1 test recall (%)")
    ax.set_ylabel("Predicted tail own loss (BA pp)")
    ax.legend(fontsize=7)
    _save(fig, dest)
