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
ENCODERS = {"virchow2": ("Virchow2", "#1f77b4"), "uni2h": ("UNI2-h", "#d62728")}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _interval(
    ax: Any, est: dict[str, float], y: float, color: str, lw: float = 1.5
) -> None:
    ax.plot([est["ci_2_5"], est["ci_97_5"]], [y, y], color=color, lw=lw)
    ax.plot(est["point"], y, "o", color=color, ms=3.5)


def _method_yticks(ax: Any) -> None:
    ax.set_yticks(
        [-i for i in range(len(METHODS))],
        [METHOD_LABELS[m] for m in METHODS],
        fontsize=7,
    )


def _save(fig: Any, name: str, handles: list[Any] | None = None) -> None:
    """Optionally add a two-column encoder legend above the panels, then write ``report/<name>``."""
    if handles:
        fig.legend(
            handles=handles,
            fontsize=7,
            frameon=False,
            ncol=2,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.07),
        )
    fig.tight_layout()
    fig.savefig(REPORT / name, bbox_inches="tight")
    plt.close(fig)


def recovery_figure(data: dict[str, dict[str, Any]]) -> None:
    """Recovery R_m per method and arm, both encoders side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3), sharey=True)
    for ax, arm in zip(axes, ARMS):
        ax.axvline(0, color="0.4", lw=0.8)
        ax.axvline(1, color="0.4", lw=0.8, ls=":")
        for i, m in enumerate(METHODS):
            for offset, (key, (_, color)) in zip((-0.15, 0.15), ENCODERS.items()):
                est = data[key]["arms"][arm]["estimates"][f"recovery_{m}"]
                _interval(ax, est, -i + offset, color)
        ax.set_title(ARM_TITLES[arm], fontsize=9)
        ax.set_xlabel("Recovery R (pp)", fontsize=8)
        ax.tick_params(labelsize=7)
    _method_yticks(axes[0])
    handles = [
        Line2D([], [], color=c, marker="o", ms=3.5, label=n)
        for n, c in ENCODERS.values()
    ]
    _save(fig, "fig_recovery.pdf", handles)


def contrast_figure(uni: dict[str, Any]) -> None:
    """Paired encoder contrast delta R_m (UNI2-h minus Virchow2); grey band marks +-1 pp."""
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3), sharey=True)
    for ax, arm in zip(axes, ARMS):
        ax.axvspan(-1, 1, color="0.9", zorder=0)
        ax.axvline(0, color="0.4", lw=0.8)
        delta = uni["encoder_contrast"][arm]["delta"]
        for i, m in enumerate(METHODS):
            _interval(ax, delta[f"recovery_{m}"], -i, "k")
        ax.set_title(ARM_TITLES[arm], fontsize=9)
        ax.set_xlabel(r"$\delta R$, UNI2-h minus Virchow2 (pp)", fontsize=8)
        ax.tick_params(labelsize=7)
    _method_yticks(axes[0])
    _save(fig, "fig_contrast.pdf")


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
        ax.set_xticks(
            range(len(labels)),
            [METHOD_LABELS[m] for m in labels],
            rotation=60,
            ha="right",
            fontsize=6,
        )
        ax.set_title(rank, fontsize=9)
        ax.tick_params(axis="y", labelsize=7)
    axes[0].set_ylabel("Recall (%)", fontsize=8)
    _save(fig, "fig_rank_recall.pdf", axes[0].get_legend_handles_labels()[0])


def main() -> None:
    """Write fig_recovery.pdf, fig_contrast.pdf and fig_rank_recall.pdf into ``report/``."""
    data = {"virchow2": _load(VIRCHOW), "uni2h": _load(UNI)}
    recovery_figure(data)
    contrast_figure(data["uni2h"])
    rank_recall_figure(data)


if __name__ == "__main__":
    main()
