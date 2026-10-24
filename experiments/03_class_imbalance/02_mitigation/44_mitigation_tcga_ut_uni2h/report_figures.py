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
BALANCED = ("cuda", "gcl", "gcl2", "crt", "disalign")
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
CONTRAST_GROUPS = (
    ("CE", (("ba_r1_ce", "BA, balanced"), ("damage", r"Damage $D_{100}$"))),
    (r"$R_m$", tuple((f"recovery_{m}", METHOD_LABELS[m]) for m in METHODS)),
    (r"$G_m$", tuple((f"gain_{m}", METHOD_LABELS[m]) for m in BALANCED)),
    (
        r"$R_m-G_m$",
        tuple((f"specific_recovery_{m}", METHOD_LABELS[m]) for m in BALANCED),
    ),
)
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
    """Left: R_m at r100 per encoder. Right: R_m - G_m (equal to R_m where G_m = 0). Dashed: CE damage."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3), sharey=True, sharex=True)
    for ax, prefix in zip(axes, ("recovery", "specific_recovery")):
        ax.axvline(0, color="0.4", lw=0.8)
        ax.axvline(1, color="0.4", lw=0.8, ls=":")
        for offset, (key, (_, color)) in zip((-0.15, 0.15), ENCODERS.items()):
            estimates = data[key]["arms"]["r100"]["estimates"]
            ax.axvline(estimates["damage"]["point"], color=color, lw=0.8, ls="--")
            for i, m in enumerate(METHODS):
                est = estimates.get(f"{prefix}_{m}", estimates[f"recovery_{m}"])
                _interval(ax, est, -i + offset, color)
        ax.tick_params(labelsize=7)
    axes[0].set_xlabel("Recovery $R$ (pp)", fontsize=8)
    axes[1].set_xlabel("Imbalance-specific recovery $R-G$ (pp)", fontsize=8)
    axes[0].set_yticks(
        [-i for i in range(len(METHODS))],
        [METHOD_LABELS[m] for m in METHODS],
        fontsize=7,
    )
    _encoder_legend(fig)
    _save(fig, "fig_recovery.pdf")


def _contrast_row(ax: Any, est: dict[str, Any], y: float) -> None:
    """One delta interval: red if encoder-dependent, hollow marker if its CI contains 0."""
    color = "#d62728" if est["encoder_dependent"] else "k"
    _interval(ax, est, y, color)
    if est["ci_2_5"] <= 0 <= est["ci_97_5"]:
        ax.plot(est["point"], y, "o", color=color, mfc="white", ms=3.5)


def contrast_figure(data: dict[str, dict[str, Any]]) -> None:
    """All pre-specified paired contrasts delta X with the +-1 pp band, grouped by quantity."""
    delta = data["uni2h"]["encoder_contrast"]["r100"]["delta"]
    fig, ax = plt.subplots(figsize=(4.6, 5.2))
    ax.axvspan(-1, 1, color="0.9", zorder=0)
    ax.axvline(0, color="0.4", lw=0.8)
    y, ticks, labels = 0.0, [], []
    for title, rows in CONTRAST_GROUPS:
        ax.text(-2.05, y + 0.1, title, fontsize=7, va="center")
        for key, label in rows:
            y -= 1
            _contrast_row(ax, delta[key], y)
            ticks.append(y)
            labels.append(label)
        y -= 1.4
    ax.set_yticks(ticks, labels, fontsize=7)
    ax.set_xlim(-2.1, 2.1)
    ax.set_ylim(y + 0.6, 0.8)
    ax.tick_params(axis="x", labelsize=7)
    ax.set_xlabel(r"$\delta X$, UNI2-h minus Virchow2 (pp)", fontsize=8)
    _save(fig, "fig_contrast.pdf")


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
    """Write fig_recovery.pdf, fig_contrast.pdf and fig_rank_recall.pdf into ``report/``."""
    data = {"virchow2": _load(VIRCHOW), "uni2h": _load(UNI)}
    recovery_figure(data)
    contrast_figure(data)
    rank_recall_figure(data)


if __name__ == "__main__":
    main()
