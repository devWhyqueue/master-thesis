"""Figures: tail-class damage per class and the per-class rank-response heatmap."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from prevalence.figures import _save

from assignment import RATIOS

__all__ = ["damage_figure", "rank_response_figure"]

_RATIO_COLOR = {10: "tab:blue", 100: "tab:red"}


def damage_figure(dists: dict[str, np.ndarray], order: list[str], dest: Path) -> None:
    """D_rho(c) per class (both ratios), classes ordered by r1 recall (low to high)."""
    fig, ax = plt.subplots(figsize=(6, 3.8), dpi=200)
    xs = np.arange(len(order))
    width = 0.3
    for i, rho in enumerate(RATIOS):
        vals = [dists[f"D_{rho}_{c}"] for c in order]
        point = np.array([v[0] for v in vals])
        lo, hi = zip(*(np.percentile(v[1:], [2.5, 97.5]) for v in vals))
        offset = (i - 0.5) * width
        ax.errorbar(
            xs + offset,
            point,
            yerr=[point - np.array(lo), np.array(hi) - point],
            fmt="o",
            color=_RATIO_COLOR[rho],
            label=rf"$\rho$ = {rho}",
        )
    ax.axhline(0.0, color="black", lw=0.8, alpha=0.5)
    ax.set_xticks(xs, order)
    ax.set_xlabel("Class (ordered by r1 recall)")
    ax.set_ylabel("BA change vs. r1 with class in the tail (pp)")
    ax.legend(fontsize=7)
    _save(fig, dest)


def rank_response_figure(
    matrix: np.ndarray, names: list[str], rho: int, dest: Path
) -> None:
    """Own-recall change vs. r1, per class (rows) x assigned rank (columns), at one ratio."""
    fig, ax = plt.subplots(figsize=(5, 4.5), dpi=200)
    bound = float(np.abs(matrix).max()) or 1.0
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-bound, vmax=bound)
    ax.set_xticks(range(matrix.shape[1]), [str(r) for r in range(matrix.shape[1])])
    ax.set_yticks(range(matrix.shape[0]), names)
    ax.set_xlabel("Assigned rank (0 = head, 6 = tail)")
    ax.set_ylabel("Class")
    ax.set_title(rf"$\rho$ = {rho}")
    fig.colorbar(im, ax=ax, label="Own recall change vs. r1 (pp)")
    _save(fig, dest)
