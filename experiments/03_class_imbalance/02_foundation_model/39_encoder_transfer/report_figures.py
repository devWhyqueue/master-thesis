"""Build the exp-39 report's figures from the report-local analysis JSON copies."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from report_assets import DATASETS, ENCODERS, REPORT, load

COLORS = {"virchow2": "#1f77b4", "uni2h": "#d62728"}
ARMS = ("B", "R10", "R100")


def _arm_panel(ax: Any, est: dict[str, Any], title: str) -> None:
    """Balanced accuracy of B/R10/R100 with 95% intervals, both encoders side by side."""
    for offset, m in zip((-0.12, 0.12), ENCODERS):
        vals = [est[f"arm_{m}_{a}"] for a in ARMS]
        pts = np.array([v["point"] for v in vals])
        err = np.array([[p - v["ci_2_5"], v["ci_97_5"] - p] for p, v in zip(pts, vals)])
        ax.errorbar(
            np.arange(3) + offset,
            pts,
            yerr=err.T,
            fmt="o-",
            ms=4,
            capsize=2,
            color=COLORS[m],
            label=ENCODERS[m],
        )
    ax.set_xticks(range(3), ARMS)
    ax.set_title(title, fontsize=9)
    ax.set_ylabel("Balanced accuracy (%)", fontsize=8)
    ax.tick_params(labelsize=7)


def _contrast_panel(ax: Any, data: dict[str, dict[str, Any]]) -> None:
    """delta_R per dataset and ratio; thick 95%, thin 97.5% (primary only)."""
    ax.axvspan(-1, 1, color="0.9", zorder=0)
    ax.axvline(0, color="0.4", lw=0.8)
    cells = [(ds, rho) for ds in DATASETS for rho in (100, 10)]
    for i, (ds, rho) in enumerate(cells):
        est = data[ds]["analysis"]["estimates"][f"delta_DR_{rho}"]
        if rho == 100:
            ax.plot([est["ci_1_25"], est["ci_98_75"]], [-i, -i], color="k", lw=1)
        width = 3 if rho == 100 else 1.5
        ax.plot([est["ci_2_5"], est["ci_97_5"]], [-i, -i], color="k", lw=width)
        ax.plot(est["point"], -i, "o", color="k", ms=4)
    labels = [f"{DATASETS[ds]} $\\rho$={rho}" for ds, rho in cells]
    ax.set_yticks([-i for i in range(len(cells))], labels, fontsize=7)
    ax.set_xlabel(r"$\delta_R$, UNI2-h minus Virchow2 (pp)", fontsize=8)
    ax.tick_params(axis="x", labelsize=7)
    ax.set_title("Encoder damage contrast", fontsize=9)


def accuracy_figure(data: dict[str, dict[str, Any]]) -> None:
    """Absolute accuracy per encoder and dataset next to the encoder damage contrasts."""
    fig, axes = plt.subplots(
        1, 3, figsize=(7.2, 2.7), gridspec_kw={"width_ratios": [1, 1, 1.25]}
    )
    for ax, ds in zip(axes[:2], DATASETS):
        _arm_panel(ax, data[ds]["analysis"]["estimates"], DATASETS[ds])
    axes[0].legend(fontsize=7, frameon=False)
    _contrast_panel(axes[2], data)
    fig.tight_layout()
    fig.savefig(REPORT / "fig_accuracy_damage.pdf", bbox_inches="tight")
    plt.close(fig)


def _class_panel(ax: Any, rec: dict[str, Any], title: str, annotate: bool) -> None:
    """Per-class recall loss B minus R100, Virchow2 (x) against UNI2-h (y)."""
    names = sorted(rec["virchow2"]["B"])
    x, y = (
        np.array([rec[m]["B"][n] - rec[m]["R100"][n] for n in names]) for m in ENCODERS
    )
    lim = [min(0.0, x.min(), y.min()) - 1, max(x.max(), y.max()) + 1]
    ax.plot(lim, lim, color="0.6", lw=0.8, ls="--")
    ax.scatter(x, y, s=12, color="0.2")
    if annotate:
        for n, xi, yi in zip(names, x, y):
            ax.annotate(
                n, (xi, yi), fontsize=6, xytext=(3, 2), textcoords="offset points"
            )
    ax.set(xlim=lim, ylim=lim)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Virchow2 recall loss B$-$R100 (pp)", fontsize=8)
    ax.set_ylabel("UNI2-h recall loss B$-$R100 (pp)", fontsize=8)
    ax.tick_params(labelsize=7)


def per_class_figure(data: dict[str, dict[str, Any]]) -> None:
    """Per-class recall loss of both encoders, one panel per dataset."""
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.0))
    for ax, ds in zip(axes, DATASETS):
        rec = data[ds]["report_diagnostics"]["per_class_recall"]
        _class_panel(ax, rec, DATASETS[ds], annotate=ds == "bracs")
    fig.tight_layout()
    fig.savefig(REPORT / "fig_per_class.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Regenerate every report figure."""
    data = load()
    accuracy_figure(data)
    per_class_figure(data)


if __name__ == "__main__":
    main()
