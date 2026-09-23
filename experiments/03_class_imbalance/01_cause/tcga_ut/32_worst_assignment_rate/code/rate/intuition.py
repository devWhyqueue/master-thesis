"""Explanatory figure for the report's Design section: single pair terms of ``S`` on the rho-100
z-profile, and how the worst/mild orders assemble such terms over all 30 classes.

Reads only the committed ``report/diagnostics.json`` (orders, main confusion partner, scores), so
it runs locally without any stored fit outputs.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Arc

from permutation.model import ranked_counts, z_of_counts

from rate import RATIO_SEARCH

__all__ = ["TCGA_CODES", "score_intuition_figure"]

_G = 20
_N_CLASSES = 30
_C_COLOR = "tab:red"
_D_COLOR = "tab:blue"

TCGA_CODES: dict[str, str] = {
    "Adrenocortical_carcinoma": "ACC",
    "Bladder_Urothelial_Carcinoma": "BLCA",
    "Brain_Lower_Grade_Glioma": "LGG",
    "Breast_invasive_carcinoma": "BRCA",
    "Cervical_squamous_cell_carcinoma_and_endocervical_adenocarcinoma": "CESC",
    "Colon_adenocarcinoma": "COAD",
    "Esophageal_carcinoma": "ESCA",
    "Glioblastoma_multiforme": "GBM",
    "Head_and_Neck_squamous_cell_carcinoma": "HNSC",
    "Kidney_Chromophobe": "KICH",
    "Kidney_renal_clear_cell_carcinoma": "KIRC",
    "Kidney_renal_papillary_cell_carcinoma": "KIRP",
    "Liver_hepatocellular_carcinoma": "LIHC",
    "Lung_adenocarcinoma": "LUAD",
    "Lung_squamous_cell_carcinoma": "LUSC",
    "Mesothelioma": "MESO",
    "Ovarian_serous_cystadenocarcinoma": "OV",
    "Pancreatic_adenocarcinoma": "PAAD",
    "Pheochromocytoma_and_Paraganglioma": "PCPG",
    "Prostate_adenocarcinoma": "PRAD",
    "Rectum_adenocarcinoma": "READ",
    "Sarcoma": "SARC",
    "Skin_Cutaneous_Melanoma": "SKCM",
    "Stomach_adenocarcinoma": "STAD",
    "Testicular_Germ_Cell_Tumors": "TGCT",
    "Thymoma": "THYM",
    "Thyroid_carcinoma": "THCA",
    "Uterine_Carcinosarcoma": "UCS",
    "Uterine_Corpus_Endometrial_Carcinoma": "UCEC",
    "Uveal_Melanoma": "UVM",
}

# (title, c, rank of c, d, rank of d, how often c is predicted as d, resulting term)
_PAIR_CASES: tuple[tuple[str, str, int, str, int, str, str], ...] = (
    ("(a) High: $c$ far below $d$", "READ", 27, "COAD", 2, "often", "large"),
    ("(b) Low: adjacent", "READ", 29, "COAD", 28, "often", "small"),
    ("(c) Zero: $c$ above $d$", "READ", 2, "COAD", 27, "often", "zero"),
    ("(d) Low: rarely confused", "READ", 27, "THCA", 2, "almost never", "near zero"),
)


def _z_profile() -> np.ndarray:
    return z_of_counts(ranked_counts(RATIO_SEARCH, _G, _N_CLASSES), _G)


def _staircase(ax: Axes, z: np.ndarray, colors: dict[int, str]) -> None:
    ranks = np.arange(len(z))
    bar_colors = [colors.get(r, "lightgray") for r in range(len(z))]
    ax.bar(ranks, z, width=0.8, color=bar_colors)
    ax.axhline(0.0, color="black", lw=0.6)
    ax.set_xlim(-1, len(z))
    ax.set_xticks([0, 10, 20, 29])


def _pair_panel(
    ax: Axes, z: np.ndarray, case: tuple[str, str, int, str, int, str, str]
) -> None:
    title, c, rank_c, d, rank_d, how_often, term = case
    _staircase(ax, z, {rank_c: _C_COLOR, rank_d: _D_COLOR})
    gap = max(z[rank_d] - z[rank_c], 0.0)
    ax.annotate(
        "",
        xy=(rank_d, z[rank_d] + 0.35),
        xytext=(rank_c, max(z[rank_c], 0.0) + 0.35),
        arrowprops={"arrowstyle": "->", "color": "black", "lw": 1.0},
    )
    ax.set_title(title, fontsize=8)
    ax.text(
        0.5,
        0.02,
        f"$c$ = {c} ($r_c$ = {rank_c})\n$d$ = {d} ($r_d$ = {rank_d})\n"
        f"$w_{{c\\rightarrow d}}$: {how_often}\n"
        f"$\\max(z_{{r_d}}-z_{{r_c}},0)$ = {gap:.2f}\n"
        f"pair term: {term}",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=5.5,
    )
    ax.set_ylim(-11.0, 3.5)
    ax.set_yticks([-4, -2, 0, 2])
    ax.set_xlabel("Rank $r$", fontsize=7)
    ax.tick_params(labelsize=6)


def _active_arcs(order: list[str], partner: dict[str, str]) -> list[tuple[int, int]]:
    """(rank of c, rank of partner) for every class that sits below its main partner."""
    rank_of = {name: i for i, name in enumerate(order)}
    return [
        (rank_of[c], rank_of[partner[c]])
        for c in order
        if rank_of[c] > rank_of[partner[c]]
    ]


def _order_panel(
    ax: Axes,
    z: np.ndarray,
    order: list[str],
    partner: dict[str, str],
    title: str,
) -> None:
    _staircase(ax, z, {})
    base = float(z.max()) + 0.4
    for rank_c, rank_d in _active_arcs(order, partner):
        width = rank_c - rank_d
        ax.add_patch(
            Arc(
                ((rank_c + rank_d) / 2, base),
                width,
                width * 0.55,
                theta1=0,
                theta2=180,
                color=_C_COLOR,
                lw=1.0,
            )
        )
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels([TCGA_CODES[c] for c in order], rotation=90, fontsize=5.5)
    ax.set_ylim(-5.0, base + 0.55 * len(order) / 2 + 0.5)
    ax.set_yticks([-4, -2, 0, 2])
    ax.tick_params(axis="y", labelsize=6)
    ax.set_title(title, fontsize=8)


def score_intuition_figure(diagnostics_path: Path, dest: Path) -> None:
    """Four single-pair cases (top) and the worst/mild orders as arcs to main partners (bottom)."""
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    z = _z_profile()
    fig = plt.figure(figsize=(7.4, 5.6), dpi=200)
    grid = fig.add_gridspec(2, 4, height_ratios=(1.0, 1.25), hspace=0.5, wspace=0.3)
    for i, case in enumerate(_PAIR_CASES):
        ax = fig.add_subplot(grid[0, i])
        _pair_panel(ax, z, case)
        if i == 0:
            ax.set_ylabel("$z_r=\\log_2(n_r/640)$", fontsize=7)
    for i, key in enumerate(("worst", "mild")):
        ax = fig.add_subplot(grid[1, 2 * i : 2 * i + 2])
        _order_panel(
            ax,
            z,
            diagnostics["orders"][key],
            diagnostics["confusion_partner"],
            f"{key.capitalize()} order, $S$ = {diagnostics['score'][key]:.0f}",
        )
        if i == 0:
            ax.set_ylabel("$z_r$", fontsize=7)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, bbox_inches="tight")
    plt.close(fig)
