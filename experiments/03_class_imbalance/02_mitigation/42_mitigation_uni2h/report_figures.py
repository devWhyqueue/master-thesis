"""Build the exp-42 report's figures from the exp-41 (Virchow2) and exp-42 (UNI2-h) analysis.json files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
REPORT = HERE / "report"
VIRCHOW = HERE.parent / "41_mitigation_bracs" / "report" / "analysis.json"
UNI = REPORT / "analysis.json"
ARMS = ("r50", "r100", "N")
METHODS = ("bs", "cuda", "la", "gcl", "posthoc_la", "crt", "disalign", "gcl2")
ENCODERS = {"virchow2": ("Virchow2", "#1f77b4"), "uni2h": ("UNI2-h", "#d62728")}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _interval(
    ax: Any, est: dict[str, float], y: float, color: str, lw: float = 1.5
) -> None:
    ax.plot([est["ci_2_5"], est["ci_97_5"]], [y, y], color=color, lw=lw)
    ax.plot(est["point"], y, "o", color=color, ms=3.5)


def recovery_figure(data: dict[str, dict[str, Any]]) -> None:
    """Recovery R_m per method and arm, both encoders side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3), sharey=True)
    for ax, arm in zip(axes, ARMS):
        ax.axvline(0, color="0.4", lw=0.8)
        for i, m in enumerate(METHODS):
            for offset, (key, (_, color)) in zip((-0.15, 0.15), ENCODERS.items()):
                est = data[key]["arms"][arm]["estimates"][f"recovery_{m}"]
                _interval(ax, est, -i + offset, color)
        ax.set_title(arm, fontsize=9)
        ax.set_xlabel("Recovery R (pp)", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0].set_yticks([-i for i in range(len(METHODS))], METHODS, fontsize=7)
    handles = [
        Line2D([], [], color=c, marker="o", ms=3.5, label=n)
        for n, c in ENCODERS.values()
    ]
    axes[0].legend(handles=handles, fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(REPORT / "fig_recovery.pdf", bbox_inches="tight")
    plt.close(fig)


def contrast_figure(uni: dict[str, Any]) -> None:
    """Paired encoder contrast delta R_m (UNI2-h minus Virchow2); grey band marks +-1 pp."""
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3), sharey=True)
    for ax, arm in zip(axes, ARMS):
        ax.axvspan(-1, 1, color="0.9", zorder=0)
        ax.axvline(0, color="0.4", lw=0.8)
        delta = uni["encoder_contrast"][arm]["delta"]
        for i, m in enumerate(METHODS):
            _interval(ax, delta[f"recovery_{m}"], -i, "k")
        ax.set_title(arm, fontsize=9)
        ax.set_xlabel(r"$\delta R$, UNI2-h minus Virchow2 (pp)", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0].set_yticks([-i for i in range(len(METHODS))], METHODS, fontsize=7)
    fig.tight_layout()
    fig.savefig(REPORT / "fig_contrast.pdf", bbox_inches="tight")
    plt.close(fig)


def rank_recall_figure(data: dict[str, dict[str, Any]]) -> None:
    """Head/body/tail recall (%) at r100 for the CE anchors and each method, per encoder."""
    labels = ("r1_ce", "r100_ce") + METHODS
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8), sharey=True)
    for ax, rank in zip(axes, ("head", "body", "tail")):
        for offset, (key, (name, color)) in zip((-0.2, 0.2), ENCODERS.items()):
            recall = data[key]["arms"]["r100"]["rank_recall"]
            ax.bar(
                np.arange(len(labels)) + offset,
                [recall[m][rank] for m in labels],
                0.4,
                color=color,
                label=name,
            )
        ax.set_xticks(range(len(labels)), labels, rotation=60, ha="right", fontsize=6)
        ax.set_title(rank, fontsize=9)
        ax.tick_params(axis="y", labelsize=7)
    axes[0].set_ylabel("Recall (%)", fontsize=8)
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(REPORT / "fig_rank_recall.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Write fig_recovery.pdf, fig_contrast.pdf and fig_rank_recall.pdf into ``report/``."""
    data = {"virchow2": _load(VIRCHOW), "uni2h": _load(UNI)}
    recovery_figure(data)
    contrast_figure(data["uni2h"])
    rank_recall_figure(data)


if __name__ == "__main__":
    main()
