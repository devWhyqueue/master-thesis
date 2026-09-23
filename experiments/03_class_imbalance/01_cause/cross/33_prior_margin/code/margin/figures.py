"""Figures: the D_sim-vs-observed-P onset curve, the margin CDF intuition figure, and H3's
subset scatter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from prevalence.figures import _save

from margin import RATIOS_NEW

__all__ = ["onset_figure", "margin_cdf_figure", "subset_scatter_figure"]


def _band(
    ax: Any, xs: list[float], d: list[np.ndarray], fmt: str, color: str, label: str
) -> None:
    lo, hi = zip(*(np.percentile(v[1:], [2.5, 97.5]) for v in d))
    ax.plot(xs, [v[0] for v in d], fmt, color=color, label=label)
    ax.fill_between(xs, lo, hi, color=color, alpha=0.15, linewidth=0)


def onset_figure(dists: dict[str, np.ndarray], dest: Path) -> None:
    """D_sim(rho) (injected) vs. the observed P(rho) (fitted), nominal ratio on a log2 x-axis."""
    xs = [float(np.log2(r)) for r in RATIOS_NEW]
    fig, ax = plt.subplots(figsize=(5.5, 3.8), dpi=200)
    _band(
        ax,
        xs,
        [dists[f"D_sim_{r}"] for r in RATIOS_NEW],
        "o-",
        "tab:red",
        "D_sim (injected)",
    )
    _band(
        ax,
        xs,
        [dists[f"observed_P_{r}"] for r in RATIOS_NEW],
        "s--",
        "tab:blue",
        "observed P (fitted)",
    )
    ax.set_xticks(xs, [str(r) for r in RATIOS_NEW])
    ax.set_xlabel(r"Imbalance ratio $\rho$ (log scale)")
    ax.set_ylabel("BA drop vs. r1 (pp)")
    ax.legend(fontsize=8)
    _save(fig, dest)


def margin_cdf_figure(
    margins_by_dataset: dict[str, np.ndarray], rhos: tuple[int, ...], dest: Path
) -> None:
    """CDF of the r1 test-patch margin (nats), with ln(rho) marked for each rho in ``rhos``."""
    fig, ax = plt.subplots(figsize=(5, 3.6), dpi=200)
    colors = plt.get_cmap("tab10")(np.linspace(0.0, 1.0, len(margins_by_dataset)))
    for (name, margins), color in zip(margins_by_dataset.items(), colors):
        xs = np.sort(margins)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        ax.plot(xs, ys, color=color, label=name)
    for r in rhos:
        ax.axvline(np.log(r), color="gray", linestyle=":", linewidth=1)
        ax.text(np.log(r), 0.02, f"ln {r}", rotation=90, fontsize=7, color="gray")
    ax.set_xlabel(r"Margin $\log p_{true} - \max_{d \neq true} \log p_d$ (nats)")
    ax.set_ylabel("CDF")
    ax.legend(fontsize=8)
    _save(fig, dest)


def _scatter_panel(
    ax: Any,
    random_pts: list[dict[str, Any]],
    greedy: dict[str, Any],
    bracs_point: tuple[float, float, float] | None,
    key: str,
    xlabel: str,
) -> None:
    """One panel: D_sim vs. ``key`` for the random subsets, the greedy one, and BRACS."""
    ax.scatter(
        [s[key] for s in random_pts],
        [s["d_sim"] for s in random_pts],
        s=10,
        alpha=0.5,
        color="tab:gray",
        label="random 7-class subset",
    )
    ax.scatter(
        [greedy[key]],
        [greedy["d_sim"]],
        s=40,
        color="tab:red",
        marker="D",
        label="greedy confused",
    )
    if bracs_point is not None:
        bx = bracs_point[0] if key == "ba" else bracs_point[2]
        ax.scatter(
            [bx], [bracs_point[1]], s=60, color="tab:blue", marker="*", label="BRACS"
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("D_sim(rho=100) (pp)")


def subset_scatter_figure(
    subsets: list[dict[str, Any]],
    bracs_point: tuple[float, float, float] | None,
    dest: Path,
) -> None:
    """D_sim vs. the subset's restricted BA (left) and median margin (right), 500 random + greedy."""
    random_pts = [s for s in subsets if s["tag"] == "random"]
    greedy = next(s for s in subsets if s["tag"] == "greedy_confused")
    fig, (ax_ba, ax_margin) = plt.subplots(1, 2, figsize=(9, 3.8), dpi=200)
    _scatter_panel(
        ax_ba, random_pts, greedy, bracs_point, "ba", "Subset balanced BA at r1 (%)"
    )
    _scatter_panel(
        ax_margin,
        random_pts,
        greedy,
        bracs_point,
        "median_margin",
        "Subset median margin (nats)",
    )
    ax_ba.legend(fontsize=7)
    _save(fig, dest)
