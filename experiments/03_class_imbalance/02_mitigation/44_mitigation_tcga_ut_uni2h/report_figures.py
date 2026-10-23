"""Build the exp-44 report's figures from the exp-43 (Virchow2) and exp-44 (UNI2-h) analysis.json files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
REPORT = HERE / "report"
VIRCHOW = HERE.parent / "43_mitigation_tcga_ut" / "report" / "analysis.json"
UNI = REPORT / "analysis.json"
METHODS = ("cuda", "la", "posthoc_la", "bs", "gcl", "disalign", "gcl2", "crt")
METHOD_LABELS = {
    "bs": "Balanced sampling",
    "cuda": "CUDA",
    "la": "Logit adj. (train-time)",
    "gcl": "GCL",
    "posthoc_la": "Logit adj. (post hoc)†",
    "crt": "cRT†",
    "disalign": "DisAlign†",
    "gcl2": "GCL†",
    "r1_ce": "CE, balanced",
    "r100_ce": r"CE, $\rho=100$",
}
ENCODERS = {"virchow2": ("Virchow2", "#1f77b4"), "uni2h": ("UNI2-h", "#d62728")}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _interval(ax: Any, est: dict[str, float], y: float, color: str) -> None:
    ax.plot([est["ci_2_5"], est["ci_97_5"]], [y, y], color=color, lw=1.5)
    ax.plot(est["point"], y, "o", color=color, ms=3.5)


def _encoder_legend(fig: Any, handles: list[Any] | None = None) -> None:
    """Two-column encoder legend above the panels; defaults to interval markers."""
    handles = handles or [
        Line2D([], [], color=c, marker="o", ms=3.5, label=n)
        for n, c in ENCODERS.values()
    ]
    fig.legend(
        handles=handles,
        fontsize=7,
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.07),
    )


def _save(fig: Any, name: str) -> None:
    fig.tight_layout()
    fig.savefig(REPORT / name, bbox_inches="tight")
    plt.close(fig)


def recovery_figure(data: dict[str, dict[str, Any]]) -> None:
    """Left: R_m at r100 per encoder, dashed lines mark each encoder's CE damage. Right: paired delta R_m."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.2, 3), sharey=True)
    left.axvline(0, color="0.4", lw=0.8)
    left.axvline(1, color="0.4", lw=0.8, ls=":")
    for offset, (key, (_, color)) in zip((-0.15, 0.15), ENCODERS.items()):
        estimates = data[key]["arms"]["r100"]["estimates"]
        left.axvline(estimates["damage"]["point"], color=color, lw=0.8, ls="--")
        for i, m in enumerate(METHODS):
            _interval(left, estimates[f"recovery_{m}"], -i + offset, color)
    left.set_xlabel("Recovery R (pp)", fontsize=8)
    right.axvspan(-1, 1, color="0.9", zorder=0)
    right.axvline(0, color="0.4", lw=0.8)
    delta = data["uni2h"]["encoder_contrast"]["r100"]["delta"]
    for i, m in enumerate(METHODS):
        _interval(right, delta[f"recovery_{m}"], -i, "k")
    right.set_xlabel(r"$\delta R$, UNI2-h minus Virchow2 (pp)", fontsize=8)
    for ax in (left, right):
        ax.tick_params(labelsize=7)
    left.set_yticks(
        [-i for i in range(len(METHODS))],
        [METHOD_LABELS[m] for m in METHODS],
        fontsize=7,
    )
    _encoder_legend(fig)
    _save(fig, "fig_recovery.pdf")


def _rank_bars(
    ax: Any, data: dict[str, dict[str, Any]], labels: tuple[str, ...], rank: str
) -> None:
    """One bar per label and encoder: r100 recall (%) of the given rank third."""
    for offset, (key, (name, color)) in zip((-0.2, 0.2), ENCODERS.items()):
        recall = data[key]["arms"]["r100"]["rank_recall"]
        ax.bar(
            np.arange(len(labels)) + offset,
            [recall[m][rank] for m in labels],
            0.4,
            color=color,
            label=name,
        )


def rank_recall_figure(data: dict[str, dict[str, Any]]) -> None:
    """Head and tail recall (%) at r100 for the CE anchors and each method, per encoder."""
    labels = ("r1_ce", "r100_ce") + METHODS
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    for ax, rank in zip(axes, ("head", "tail")):
        _rank_bars(ax, data, labels, rank)
        ax.set_xticks(
            range(len(labels)),
            [METHOD_LABELS[m] for m in labels],
            rotation=60,
            ha="right",
            fontsize=6,
        )
        ax.set_ylim(50, 82)
        ax.set_title(f"{rank} third", fontsize=9)
        ax.tick_params(axis="y", labelsize=7)
    axes[0].set_ylabel("Recall (%)", fontsize=8)
    _encoder_legend(fig, axes[0].get_legend_handles_labels()[0])
    _save(fig, "fig_rank_recall.pdf")


def main() -> None:
    """Write fig_recovery.pdf and fig_rank_recall.pdf into ``report/``."""
    data = {"virchow2": _load(VIRCHOW), "uni2h": _load(UNI)}
    recovery_figure(data)
    rank_recall_figure(data)


if __name__ == "__main__":
    main()
