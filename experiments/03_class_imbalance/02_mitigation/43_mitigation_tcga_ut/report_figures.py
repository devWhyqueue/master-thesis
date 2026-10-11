"""Build the exp-43 report's figures from its analysis.json."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

REPORT = Path(__file__).resolve().parent / "report"
ARMS = ("r50", "r100", "N")
METHODS = ("la", "posthoc_la", "cuda", "gcl", "disalign", "bs", "crt", "gcl2")
ARM_TITLES = {"r50": r"$\rho=50$", "r100": r"$\rho=100$", "N": "Native shares"}
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
RANKS = {"head": ("#1f77b4", "s"), "body": ("0.45", "o"), "tail": ("#d62728", "^")}


def _yticks(ax: Any, labels: tuple[str, ...]) -> None:
    ax.set_yticks(
        [-i for i in range(len(labels))], [METHOD_LABELS[m] for m in labels], fontsize=7
    )


def recovery_figure(analysis: dict[str, Any]) -> None:
    """Paired recovery R_m per method and arm; grey line and band mark the CE damage D."""
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3), sharey=True)
    for ax, arm in zip(axes, ARMS):
        estimates = analysis["arms"][arm]["estimates"]
        damage = estimates["damage"]
        ax.axvspan(damage["ci_2_5"], damage["ci_97_5"], color="0.88", zorder=0)
        ax.axvline(damage["point"], color="0.55", lw=1)
        ax.axvline(0, color="0.2", lw=0.8)
        ax.axvline(1, color="0.2", lw=0.8, ls=":")
        for i, m in enumerate(METHODS):
            est = estimates[f"recovery_{m}"]
            ax.plot([est["ci_2_5"], est["ci_97_5"]], [-i, -i], color="k", lw=1.5)
            ax.plot(est["point"], -i, "o", color="k", ms=3.5)
        ax.set_title(ARM_TITLES[arm], fontsize=9)
        ax.set_xlabel("Recovery R (pp)", fontsize=8)
        ax.tick_params(labelsize=7)
    _yticks(axes[0], METHODS)
    fig.tight_layout()
    fig.savefig(REPORT / "fig_recovery.pdf", bbox_inches="tight")
    plt.close(fig)


def rank_recall_figure(analysis: dict[str, Any]) -> None:
    """Head/body/tail recall (%) at r100 for the CE references and each method."""
    labels = ("r1_ce", "r100_ce") + METHODS
    recall = analysis["arms"]["r100"]["rank_recall"]
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for i, m in enumerate(labels):
        values = [recall[m][rank] for rank in RANKS]
        ax.plot([min(values), max(values)], [-i, -i], color="0.8", lw=1, zorder=0)
    for rank, (color, marker) in RANKS.items():
        ax.plot(
            [recall[m][rank] for m in labels],
            [-i for i in range(len(labels))],
            marker,
            color=color,
            ms=4.5,
            label=rank,
        )
    _yticks(ax, labels)
    ax.set_xlabel("Recall (%)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(
        fontsize=7, frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1)
    )
    fig.tight_layout()
    fig.savefig(REPORT / "fig_rank_recall.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Write fig_recovery.pdf and fig_rank_recall.pdf into ``report/``."""
    analysis = json.loads((REPORT / "analysis.json").read_text(encoding="utf-8"))
    recovery_figure(analysis)
    rank_recall_figure(analysis)


if __name__ == "__main__":
    main()
