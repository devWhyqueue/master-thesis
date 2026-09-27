"""The recovery figure: selected BA per method against the r1/r100 CE anchors."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

__all__ = ["recovery_figure"]


def _error_bars(
    estimates: dict[str, dict[str, float]], methods: tuple[str, ...]
) -> tuple[list[float], list[float], list[float]]:
    points = [estimates[f"selected_{m}"]["point"] for m in methods]
    lo = [
        estimates[f"selected_{m}"]["point"] - estimates[f"selected_{m}"]["ci_2_5"]
        for m in methods
    ]
    hi = [
        estimates[f"selected_{m}"]["ci_97_5"] - estimates[f"selected_{m}"]["point"]
        for m in methods
    ]
    return points, lo, hi


def _anchor_lines(ax: Any, estimates: dict[str, dict[str, float]]) -> None:
    ax.axhline(
        estimates["arm_r1_ce"]["point"],
        color="tab:green",
        ls="--",
        lw=1,
        label="r1 CE (undamaged)",
    )
    ax.axhline(
        estimates["arm_r100_ce"]["point"],
        color="tab:red",
        ls="--",
        lw=1,
        label="r100 CE (damage anchor)",
    )


def recovery_figure(
    estimates: dict[str, dict[str, float]], methods: tuple[str, ...], dest: Path
) -> None:
    """Selected BA per method with its 95% interval, against the r1/r100 CE anchor lines."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 3.6), dpi=200)
    xs = np.arange(len(methods))
    points, lo, hi = _error_bars(estimates, methods)
    ax.errorbar(xs, points, yerr=[lo, hi], fmt="o", color="tab:blue", capsize=3)
    _anchor_lines(ax, estimates)
    ax.set_xticks(xs, list(methods), rotation=30, ha="right")
    ax.set_ylabel("Patient-macro BA (%)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(dest)
    plt.close(fig)
