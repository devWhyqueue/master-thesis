"""Imbalance figure: balanced-correct predictions lost and balanced errors fixed per arm."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from gap.plots.style import DATASETS, GAINED, INK, LABEL, LOST, save

__all__ = ["imbalance_figure"]

_ARMS = (("P100", "frequency only"), ("S100", "support only"), ("r100", "ratio (both)"))


def _arm_bars(ax: Any, arm: dict[str, Any], y: int) -> None:
    """Lost (right), fixed (left), and the net-damage marker with its value."""
    lost, gained = arm["lost"] * 100, arm["gained"] * 100
    first = y == 0
    ax.barh(y, lost, color=LOST, height=0.55, label="correct lost" if first else None)
    ax.barh(
        y, -gained, color=GAINED, height=0.55, label="errors fixed" if first else None
    )
    net = lost - gained
    ax.plot(net, y, "D", color="#0b0b0b", ms=4, label="net damage" if first else None)
    ax.text(lost + 0.4, y, f"{net:.2f}", va="center", fontsize=7, color=INK)


def _panel(ax: Any, analysis: dict[str, Any], ds: str) -> None:
    for y, (arm, _) in enumerate(_ARMS):
        _arm_bars(ax, analysis[arm], y)
    ax.set_yticks(range(len(_ARMS)), [label for _, label in _ARMS])
    ax.axvline(0, color=INK, linewidth=0.8)
    ax.set_title(LABEL[ds], loc="left", fontsize=8)
    ax.set_xlabel("patient-macro share of test predictions (pp)")
    ax.invert_yaxis()


def imbalance_figure(data: dict[str, dict[str, Any]], path: Path) -> None:
    """Balanced-correct predictions lost and balanced-wrong predictions gained, per arm."""
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.0), sharex=True)
    for ax, ds in zip(axes, DATASETS):
        _panel(ax, data[ds], ds)
    handles, labels = axes[0].get_legend_handles_labels()
    legend = {"fontsize": 7, "loc": "upper center", "ncol": 3}
    fig.legend(handles, labels, bbox_to_anchor=(0.5, -0.12), **legend)
    save(fig, path)
