"""Representation figures: neighbour agreement (original dimensions) and an illustrative PCA."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from gap.plots.codes import short_name
from gap.plots.style import COLOR, DATASETS, HIGHLIGHT, LABEL, MUTED, PAIRS, save

__all__ = ["geometry_figure", "pca_figure"]

_PRIMARY_K = "k15"
_METRIC_STYLES = (("cosine", "-o"), ("euclidean", "--s"))


def _adjusted_per_class(analysis: dict[str, Any]) -> np.ndarray:
    """Split-mean chance-adjusted cosine agreement at k = 15, one value per class."""
    geo = analysis["geometry"]
    raw = np.mean([g["cosine"][_PRIMARY_K]["per_class"] for g in geo], axis=0)
    chance = 1.0 / len(analysis["class_names"])
    return (raw - chance) / (1.0 - chance)


def _class_dots(ax: Any, data: dict[str, dict[str, Any]]) -> None:
    """One dot per class and a class-mean bar per dataset."""
    for x, ds in enumerate(DATASETS):
        adj = _adjusted_per_class(data[ds])
        jitter = ((np.argsort(np.argsort(adj)) % 5) - 2) * 0.05
        ax.scatter(x + jitter, adj, s=14, color=COLOR[ds], edgecolor="white", lw=0.5)
        ax.hlines(adj.mean(), x - 0.22, x + 0.22, color="#0b0b0b", linewidth=1.5)
    ax.set_xticks(range(len(DATASETS)), [LABEL[d] for d in DATASETS])
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylabel("chance-adjusted agreement, k = 15")
    ax.set_title("(a) one dot per class, bar = class mean", loc="left", fontsize=8)


def _k_curves(ax: Any, data: dict[str, dict[str, Any]]) -> None:
    """Class-mean chance-adjusted agreement against k for both distances."""
    for ds in DATASETS:
        geo = data[ds]["geometry"]
        for metric, style in _METRIC_STYLES:
            ks = sorted(geo[0][metric], key=lambda k: int(k[1:]))
            y = [np.mean([g[metric][k]["chance_adjusted"] for g in geo]) for k in ks]
            label = f"{LABEL[ds]}, {metric}"
            x = [int(k[1:]) for k in ks]
            ax.plot(x, y, style, color=COLOR[ds], markersize=4, lw=1.5, label=label)
    ax.set_xticks([5, 15, 30])
    ax.set_xlabel("neighbours k")
    ax.set_ylabel("class-mean adjusted agreement")
    ax.set_ylim(0, None)
    ax.set_title("(b) sensitivity to k and distance", loc="left", fontsize=8)
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(0.0, 0.66))


def geometry_figure(data: dict[str, dict[str, Any]], path: Path) -> None:
    """(a) per-class chance-adjusted agreement at k=15; (b) class mean against k, both metrics."""
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(6.3, 2.6), width_ratios=(1.0, 1.25))
    _class_dots(ax_a, data)
    _k_curves(ax_b, data)
    save(fig, path)


def _pca_panel(ax: Any, analysis: dict[str, Any], ds: str) -> None:
    """Split-0 test points on that split's training-fitted axes, confusion pair highlighted."""
    geo = analysis["geometry"][0]
    pts = geo["pca_points"]
    z = np.column_stack([pts["pc1"], pts["pc2"]])
    labels = np.asarray(pts["label"])
    style = {"linewidth": 0, "rasterized": True}
    ax.scatter(*z.T, s=2, color=MUTED, alpha=0.35, label="other classes", **style)
    for name, color in zip(PAIRS[ds], HIGHLIGHT):
        mask = labels == analysis["class_names"].index(name)
        ax.scatter(
            *z[mask].T, s=3, color=color, alpha=0.6, label=short_name(name), **style
        )
    ev = geo["pca_explained_variance"]
    ax.set_xlabel(f"PC1 ({ev[0] * 100:.0f}% of variance)")
    ax.set_ylabel(f"PC2 ({ev[1] * 100:.0f}%)")
    ax.set_title(LABEL[ds], loc="left", fontsize=8)
    ax.legend(fontsize=7, markerscale=4, loc="best")
    ax.set_xticklabels([])
    ax.set_yticklabels([])


def pca_figure(data: dict[str, dict[str, Any]], path: Path) -> None:
    """Held-out split-0 test patches on each dataset's own training-fitted PCA axes."""
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.0))
    for ax, ds in zip(axes, DATASETS):
        _pca_panel(ax, data[ds], ds)
    save(fig, path)
