"""Render the controlled-imbalance construction figures and summary table.

Produces (1) a Section-3 summary of target head-to-tail ratios at patch and
slide level against the long-tail stress-test band (companion to the native
figure), (2) an appendix grid of native-vs-target distributions at both levels,
and (3) a both-levels summary table. Figures/tables for the report only; no
experiments are re-run. See ``controlled_construction.py`` for the constructions.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

from compute_imbalance import entropy_imbalance, head_tail_ratio
from controlled_construction import build_targets

logger = logging.getLogger(__name__)

COUNTS = Path(__file__).resolve().parent / "outputs" / "counts" / "counts.json"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"

NATIVE_COLOR = "#b0b0b0"
TARGET_COLOR = "#c8615a"
PATCH_COLOR = "#4c78a8"
SLIDE_COLOR = "#c8615a"
STRESS_LOW, STRESS_HIGH = 100.0, 200.0


def plot_target_regimes(targets: dict, path: Path) -> None:
    """Grouped bars of target head-to-tail ratio (patch vs slide) vs stress band."""
    names = list(targets)
    patch = [head_tail_ratio(targets[n]["patch"]["target"]) for n in names]
    slide = [head_tail_ratio(targets[n]["slide"]["target"]) for n in names]
    x, width = np.arange(len(names)), 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axhspan(
        STRESS_LOW,
        STRESS_HIGH,
        color="orange",
        alpha=0.18,
        label=r"Long-tail stress-test regime ($\rho=100$--$200$)",
    )
    ax.bar(x - width / 2, patch, width, label="Patch/tile", color=PATCH_COLOR)
    ax.bar(x + width / 2, slide, width, label="Slide", color=SLIDE_COLOR)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel(r"Target head-to-tail ratio $\rho=N_{\max}/N_{\min}$")
    ax.set_ylim(1, 300)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _bar_panel(ax, panel: dict, title: str) -> None:
    labels = panel["labels"]
    x, width = np.arange(len(labels)), 0.35
    ax.bar(x - width / 2, panel["native"], width, label="Native", color=NATIVE_COLOR)
    ax.bar(x + width / 2, panel["target"], width, label="Target", color=TARGET_COLOR)
    ax.axhline(panel["floor"], ls=":", c="k", lw=1, label="floor")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6, rotation=0)
    ax.set_title(title, fontsize=9)


def _rank_panel(ax, panel: dict, title: str) -> None:
    ranks = range(1, len(panel["native"]) + 1)
    ax.plot(ranks, panel["native"], color=NATIVE_COLOR, ls="--", label="Native")
    ax.plot(ranks, panel["target"], color=TARGET_COLOR, label="Target")
    ax.axhline(panel["floor"], ls=":", c="k", lw=1, label="floor")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Class rank", fontsize=8)
    ax.set_title(title, fontsize=9)


def _draw_panel(ax, panel: dict, title: str) -> None:
    (_rank_panel if len(panel["labels"]) > 10 else _bar_panel)(ax, panel, title)
    ax.set_ylabel("Support")
    ax.legend(fontsize=6)


def plot_distributions(targets: dict, path: Path) -> None:
    """4x2 grid of native-vs-target support: rows = datasets, cols = patch/slide."""
    fig, axes = plt.subplots(len(targets), 2, figsize=(9, 12))
    for row, (name, levels) in zip(axes, targets.items()):
        _draw_panel(row[0], levels["patch"], f"{name} --- patch")
        _draw_panel(row[1], levels["slide"], f"{name} --- slide")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _row(name: str, level: str, panel: dict) -> str:
    native, target = panel["native"], panel["target"]
    n_nat, n_tgt = sum(native), sum(target)
    pct = 100 * n_tgt / n_nat
    return (
        f"{name} & {level} & {len(native)} & {panel['method']} & "
        f"\\num{{{n_nat}}} $\\to$ \\num{{{n_tgt}}} ({pct:.0f}\\%) & "
        f"{head_tail_ratio(native):.1f}:1 $\\to$ {head_tail_ratio(target):.1f}:1 & "
        f"\\num{{{entropy_imbalance(native):.3f}}} $\\to$ "
        f"\\num{{{entropy_imbalance(target):.3f}}}\\\\"
    )


def write_targets_table(targets: dict, path: Path) -> None:
    """Write the both-levels construction summary as a booktabs LaTeX table."""
    header = (
        "Dataset & Level & $K$ & Method & Support & $\\rho$ & $1-H_{\\mathrm{norm}}$"
    )
    body = ["\\begin{tabular}{llrllll}", "\\toprule", f"{header}\\\\", "\\midrule"]
    for i, (name, levels) in enumerate(targets.items()):
        if i:
            body.append("\\addlinespace")
        body.append(_row(name, "Patch", levels["patch"]))
        body.append(_row("", "Slide", levels["slide"]))
    body.extend(["\\bottomrule", "\\end{tabular}"])
    path.write_text("\n".join(body) + "\n")


def main() -> None:
    """Generate the target-regime figure, distribution grid, and targets table."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    targets = build_targets(json.loads(COUNTS.read_text(encoding="utf-8")))
    plot_target_regimes(targets, FIGURES_DIR / "controlled_imbalance_regimes.png")
    plot_distributions(targets, FIGURES_DIR / "controlled_imbalance_construction.png")
    write_targets_table(targets, TABLES_DIR / "controlled_imbalance_targets.tex")
    logger.info("Wrote figures to %s and table to %s", FIGURES_DIR, TABLES_DIR)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
