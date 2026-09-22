"""Figure: BA change vs. r1, prior-only (P) vs. support-only (S) vs. the full ratio arm (r)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from prevalence.figures import _save

from cause import RATIOS_NEW

__all__ = ["component_figure"]

_SERIES = (
    ("delta_r", "o-", "tab:red", "r (prior + support)"),
    ("delta_P", "s-", "tab:blue", "P (prior only)"),
    ("delta_S", "^-", "tab:green", "S (support only)"),
)


def _band(
    ax: Any, xs: list[float], d: list[np.ndarray], fmt: str, color: str, label: str
) -> None:
    """Point estimates joined by a line, with a shaded 95% percentile band over replicates."""
    lo, hi = zip(*(np.percentile(v[1:], [2.5, 97.5]) for v in d))
    ax.plot(xs, [v[0] for v in d], fmt, color=color, label=label)
    ax.fill_between(xs, lo, hi, color=color, alpha=0.15, linewidth=0)


def component_figure(
    dists: dict[str, np.ndarray], rho: dict[str, float], dest: Path
) -> None:
    """Delta-BA vs realized rho (log2 axis; r's realized rho shared as x for P and S), with P + S dashed."""
    xs = [float(np.log2(rho[f"r{r}"])) for r in RATIOS_NEW]
    fig, ax = plt.subplots(figsize=(5.5, 3.8), dpi=200)
    for family, fmt, color, label in _SERIES:
        _band(ax, xs, [dists[f"{family}_{r}"] for r in RATIOS_NEW], fmt, color, label)
    summed = [dists[f"delta_P_{r}"] + dists[f"delta_S_{r}"] for r in RATIOS_NEW]
    ax.plot(xs, [s[0] for s in summed], "d--", color="tab:purple", label="P + S")
    ax.set_xticks(xs, [str(r) for r in RATIOS_NEW])
    ax.set_xlabel(r"Imbalance ratio $\rho$ (log scale)")
    ax.set_ylabel("BA change vs. r1 (pp)")
    ax.legend(fontsize=7)
    _save(fig, dest)
