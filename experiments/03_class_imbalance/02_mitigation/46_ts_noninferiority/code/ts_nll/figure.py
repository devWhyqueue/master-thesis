"""Draw the primary-contrast figure (``report/fig_contrasts.pdf``) from ``analysis.json``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from ts_nll.render import METHODS, SETTINGS

__all__ = ["write_figure"]

# Decision is encoded by marker shape and fill, so it never relies on colour alone.
_MARKERS: dict[str, dict[str, Any]] = {
    "no_meaningful_gain": dict(marker="o", mfc="0.15", mec="0.15"),
    "inconclusive": dict(marker="o", mfc="white", mec="0.15"),
    "method_better": dict(marker="D", mfc="#c0392b", mec="#c0392b"),
}
_LABELS = {
    "no_meaningful_gain": "no meaningful gain",
    "inconclusive": "inconclusive",
    "method_better": "method better",
}


def _panel(ax: Any, methods: dict[str, Any], margin: float) -> None:
    ax.axvspan(-1, -margin, color="0.93", zorder=0)
    ax.axvline(0, color="0.55", lw=0.8, zorder=1)
    ax.axvline(-margin, color="0.3", lw=0.8, ls="--", zorder=1)
    for y, method in enumerate(METHODS):
        delta = methods[method]["nll_ts_delta"]
        ax.plot(
            [delta["lower"], delta["upper"]], [y, y], color="0.15", lw=1.2, zorder=2
        )
        ax.plot(
            delta["estimate"],
            y,
            ms=4.5,
            ls="none",
            zorder=3,
            **_MARKERS[delta["decision"]],
        )


def write_figure(analysis: dict[str, Any], path: Path) -> None:
    """Four panels (dataset rows, encoder columns) of delta with its Bonferroni interval and the margin."""
    arm = analysis["protocol"]["protocol"]["primary_arm"]
    margin = analysis["protocol"]["protocol"]["margin_nats"]
    fig, axes = plt.subplots(2, 2, figsize=(6.2, 4.4), sharex=True, sharey=True)
    for ax, (key, arms) in zip(axes.flat, analysis["settings"].items()):
        _panel(ax, arms[arm]["methods"], margin)
        ax.set_title(", ".join(SETTINGS[key]), fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0, 0].set_yticks(range(len(METHODS)), list(METHODS.values()))
    axes[0, 0].invert_yaxis()
    axes[0, 0].set_xlim(-0.19, 0.09)
    for ax in axes[1]:
        ax.set_xlabel(r"$\delta$ = scaled NLL, method minus CE (nats)", fontsize=8)
    _legend(fig, margin)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _legend(fig: Any, margin: float) -> None:
    """One row above the panels: the three decision markers and the margin line."""
    handles = [
        Line2D([], [], ls="none", ms=4.5, label=_LABELS[k], **v)
        for k, v in _MARKERS.items()
    ]
    handles.append(
        Line2D([], [], color="0.3", lw=0.8, ls="--", label=f"margin $-{margin}$")
    )
    fig.legend(
        handles=handles,
        fontsize=7,
        frameon=False,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
    )
