"""Figures: metric curves vs realized rho and realized class distributions per arm."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from prevalence import ARMS, RATIOS

__all__ = [
    "CALIBRATION_PANELS",
    "DISCRIMINATION_PANELS",
    "distribution_figure",
    "metric_figure",
]


def _band(ax: Any, xs: list[float], d: list[np.ndarray], fmt: str, **kw: Any) -> None:
    """Point estimates joined by a line, with a shaded 95% percentile band over replicates."""
    lo, hi = zip(*(np.percentile(v[1:], [2.5, 97.5]) for v in d))
    ax.plot(xs, [v[0] for v in d], fmt, color="tab:blue", **kw)
    ax.fill_between(xs, lo, hi, color="tab:blue", alpha=0.15, linewidth=0)


DISCRIMINATION_PANELS = ((r"BA change vs. $\rho$ = 1 (pp)", "ba", None),)
CALIBRATION_PANELS = (
    ("Macro NLL (nats)", "nll", "nll_ts"),
    ("ECE (pp)", "ece", "ece_ts"),
)


def _native_marker(
    ax: Any, x: float, native: np.ndarray, fill: str, label: str
) -> None:
    """Native arm's point estimate at its realized rho, with its 95% percentile interval."""
    lo, hi = np.percentile(native[1:], [2.5, 97.5])
    ax.errorbar(
        [x],
        [native[0]],
        yerr=[[native[0] - lo], [hi - native[0]]],
        fmt="D",
        color="tab:red",
        mfc=fill,
        zorder=5,
        label=label,
    )


def _save(fig: Any, dest: Path) -> None:
    """Write one figure and release it."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(dest)
    plt.close(fig)


def metric_figure(
    dists: dict[str, np.ndarray],
    rho: dict[str, float],
    dest: Path,
    panels: tuple[tuple[str, str, str | None], ...],
) -> None:
    """One panel per metric vs realized rho (log2 axis), 95% bands, native at its realized rho."""
    dists = dists | {f"ba_{a}": dists[f"arm_{a}"] - dists["arm_r1"] for a in ARMS}
    xs = [float(np.log2(rho[f"r{r}"])) for r in RATIOS]
    native_x = float(np.log2(rho["N"]))
    fig, axes = plt.subplots(
        1, len(panels), figsize=(5 * len(panels), 3.6), dpi=200, squeeze=False
    )
    for ax, (label, raw_key, ts_key) in zip(axes[0], panels):
        keys = (raw_key,) if ts_key is None else (raw_key, ts_key)
        for key, fmt, fill in zip(keys, ("o-", "o--"), ("tab:red", "none")):
            name = "raw" if key == raw_key else "temperature-scaled"
            name = "ratio arms" if ts_key is None else name
            _band(ax, xs, [dists[f"{key}_r{r}"] for r in RATIOS], fmt, label=name)
            native = f"native ({name})" if ts_key else "native"
            _native_marker(ax, native_x, dists[f"{key}_N"], fill, native)
        ax.set_xticks(xs, [str(r) for r in RATIOS])
        ax.set_xlabel(r"Imbalance ratio $\rho$ (log scale)")
        ax.set_ylabel(label)
        ax.legend(fontsize=7)
    _save(fig, dest)


def distribution_figure(counts: dict[str, np.ndarray], dest: Path) -> None:
    """Realized class counts by rank (log y): ratio arms on a sequential ramp, native with its range."""
    ranks = np.arange(1, counts["N"].shape[1] + 1)
    colors = plt.get_cmap("Blues")(np.linspace(0.35, 1.0, len(RATIOS)))
    fig, ax = plt.subplots(figsize=(5, 3.6), dpi=200)
    for r, color in zip(RATIOS, colors):
        ax.plot(
            ranks,
            counts[f"r{r}"].mean(0),
            "o-",
            ms=3,
            lw=1.5,
            color=color,
            label=rf"$\rho$ = {r}",
        )
    native = counts["N"]
    ax.fill_between(
        ranks, native.min(0), native.max(0), color="tab:red", alpha=0.2, lw=0
    )
    ax.plot(ranks, native.mean(0), "D--", ms=3, lw=1.5, color="tab:red", label="native")
    ax.set_yscale("log")
    ax.set_xlabel("Class rank (largest to smallest)")
    ax.set_ylabel("Training patches per class (log scale)")
    ax.legend(fontsize=7, ncol=2)
    _save(fig, dest)
