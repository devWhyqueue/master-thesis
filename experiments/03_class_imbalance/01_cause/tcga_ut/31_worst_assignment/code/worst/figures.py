"""Figures: observed damage against the score S, and per-class recall change against confusion
partner rank gap (sep vs tog)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from prevalence.figures import _save

__all__ = ["damage_vs_score_figure", "pair_gap_figure"]

_MARKER_COLOR = {
    "easy": "tab:green",
    "hard": "tab:orange",
    "sep": "tab:red",
    "flip": "tab:purple",
    "tog": "tab:blue",
}


def _plot_points(ax: Axes, points: dict[str, tuple[float, np.ndarray]]) -> None:
    """One point + 95% CI whisker per named (S, D-distribution) reference."""
    for name, (s, dist) in points.items():
        point = float(dist[0])
        lo, hi = np.percentile(dist[1:], [2.5, 97.5])
        ax.errorbar(
            [s],
            [point],
            yerr=[[point - lo], [hi - point]],
            fmt="D",
            markersize=7,
            capsize=3,
            color=_MARKER_COLOR.get(name, "black"),
            label=name,
            zorder=5,
        )


def damage_vs_score_figure(
    s_random: np.ndarray,
    d_random: np.ndarray,
    points: dict[str, tuple[float, np.ndarray]],
    dest: Path,
) -> None:
    """Observed BA damage against S: 30 random orders (grey), named orders with 95% CI."""
    fig, ax = plt.subplots(figsize=(6, 4.2), dpi=200)
    ax.scatter(
        s_random,
        d_random,
        s=14,
        color="tab:gray",
        alpha=0.6,
        label="random orders (exp-25)",
    )
    _plot_points(ax, points)
    ax.set_xlabel("S (directed pair-separation score)")
    ax.set_ylabel("BA damage vs. r1 (pp)")
    ax.legend(fontsize=7)
    _save(fig, dest)


def pair_gap_figure(
    gap_sep: list[float],
    change_sep: list[float],
    gap_tog: list[float],
    change_tog: list[float],
    dest: Path,
) -> None:
    """Per-class own-recall change vs. rank gap to its main confusion partner, sep vs tog."""
    fig, ax = plt.subplots(figsize=(5, 4), dpi=200)
    ax.scatter(gap_sep, change_sep, s=14, color="tab:red", label="sep")
    ax.scatter(gap_tog, change_tog, s=14, color="tab:blue", label="tog")
    ax.axhline(0.0, color="black", lw=0.8, alpha=0.5)
    ax.set_xlabel("Rank gap to main confusion partner")
    ax.set_ylabel("Own recall change vs. r1 (pp)")
    ax.legend(fontsize=7)
    _save(fig, dest)
