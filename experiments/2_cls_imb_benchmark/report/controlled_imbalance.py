"""Specify controlled class imbalance constructions for the benchmark datasets.

Applies research-grounded controlled-imbalance constructions --- truncated
exponential decay with a floor (Cui et al., CVPR 2019) and step imbalance
(Buda et al., 2018), plus simple ratio downsampling for the binary case --- to
each dataset's native slide counts, and visualises the resulting native-vs-
target support. See ``native_imbalance_regimes.py`` for the companion
motivation figure. This only produces figures/tables for the report; no
experiments are re-run.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.axes
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

logger = logging.getLogger(__name__)

COUNTS = (
    Path(__file__).resolve().parents[2]
    / "0_datasets"
    / "report"
    / "outputs"
    / "counts"
    / "counts.json"
)
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"

NATIVE_COLOR = "#b0b0b0"
TARGET_COLOR = "#c8615a"


def entropy_imbalance(counts: list[float]) -> float:
    """Return 1 - normalised entropy of the class-count distribution."""
    c = np.asarray([v for v in counts if v > 0], dtype=float)
    k = c.size
    if k <= 1:
        return 1.0
    p = c / c.sum()
    return float(1.0 - (-(p * np.log(p)).sum()) / np.log(k))


def head_tail_ratio(counts: list[float]) -> float:
    """Return the max:min (head-to-tail) class-support ratio."""
    positive = [v for v in counts if v > 0]
    return max(positive) / min(positive)


Counts = dict[str, int]


def exp_decay_targets(ordered: list[int], n_floor: int) -> list[int]:
    """Cui et al. truncated exponential decay: n_i = max(floor, N_max * mu**i).

    ``ordered`` must already be in the order the decay follows (native rank
    for TCGA-UT, fixed ordinal grade for PANDA). Targets are capped at the
    native count so the construction only ever downsamples.
    """
    k = len(ordered)
    n_max = ordered[0]
    mu = (n_floor / n_max) ** (1 / (k - 1))
    decay = [n_max * mu**i for i in range(k)]
    return [min(n, max(n_floor, round(d))) for n, d in zip(ordered, decay)]


def step_targets(native: Counts, head: set[str], n_floor: int) -> Counts:
    """Buda et al. step imbalance: head keeps native support, tail is floored."""
    return {c: (n if c in head else min(n, n_floor)) for c, n in native.items()}


def ratio_downsample(n_major: int, n_minor: int, ratio: float, floor: int) -> int:
    """Downsample the minority class to a fixed major:minor ratio, floored."""
    return min(n_minor, max(floor, round(n_major / ratio)))


def plot_controlled_construction(targets: dict[str, dict], path: Path) -> None:
    """2x2 grid of native vs controlled-target support per dataset."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    panels = [
        ("TCGA-UT", "TCGA-UT (exp. decay)", _plot_rank_panel),
        ("BRACS", "BRACS (step)", _plot_bar_panel),
        ("PANDA", "PANDA (ordinal exp. decay)", _plot_bar_panel),
        ("CAMELYON16", "CAMELYON16 (ratio)", _plot_bar_panel),
    ]
    for ax, (name, title, plot_fn) in zip(axes.flat, panels):
        plot_fn(ax, targets[name], title)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _plot_rank_panel(ax: matplotlib.axes.Axes, panel: dict, title: str) -> None:
    ranks = range(1, len(panel["native"]) + 1)
    ax.plot(ranks, panel["native"], color=NATIVE_COLOR, linestyle="--", label="Native")
    ax.plot(ranks, panel["target"], color=TARGET_COLOR, label="Target")
    ax.axhline(panel["floor"], ls=":", c="k", lw=1, label="$N_{floor}$")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Class rank")
    ax.set_ylabel("Slides")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7)


def _plot_bar_panel(ax: matplotlib.axes.Axes, panel: dict, title: str) -> None:
    labels = panel["labels"]
    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width / 2, panel["native"], width, label="Native", color=NATIVE_COLOR)
    ax.bar(x + width / 2, panel["target"], width, label="Target", color=TARGET_COLOR)
    ax.axhline(panel["floor"], ls=":", c="k", lw=1, label="$N_{floor}$")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Slides")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7)


def write_targets_table(targets: dict[str, dict], path: Path) -> None:
    """Write the per-dataset construction summary as a booktabs LaTeX table."""
    header = (
        "Dataset & Method & $K$ & $N_{floor}$ & $\\rho$ (native $\\to$ target) & "
        "$1-H_{\\mathrm{norm}}$ (native $\\to$ target)"
    )
    rows = [_target_row(name, panel) for name, panel in targets.items()]
    spec = "l" * (header.count("&") + 1)
    body = ["\\begin{tabular}{" + spec + "}", "\\toprule", f"{header}\\\\", "\\midrule"]
    body.extend(rows)
    body.extend(["\\bottomrule", "\\end{tabular}"])
    path.write_text("\n".join(body) + "\n")


def _target_row(name: str, panel: dict) -> str:
    native, target = panel["native"], panel["target"]
    rho_native, rho_target = head_tail_ratio(native), head_tail_ratio(target)
    h_native, h_target = entropy_imbalance(native), entropy_imbalance(target)
    return (
        f"{name} & {panel['method']} & {len(native)} & {panel['floor']} & "
        f"{rho_native:.1f}:1 $\\to$ {rho_target:.1f}:1 & "
        f"\\num{{{h_native:.3f}}} $\\to$ \\num{{{h_target:.3f}}}\\\\"
    )


def _tcga_panel(counts: dict[str, int]) -> dict:
    """TCGA-UT: exponential decay ranked by native support, floor 20."""
    native = sorted(counts.values(), reverse=True)
    floor = 20
    return {
        "method": "Exp.\\ decay",
        "native": native,
        "target": exp_decay_targets(native, floor),
        "floor": floor,
    }


def _bracs_panel(counts: dict[str, int]) -> dict:
    """BRACS: step imbalance, benign/usual head native, atypical/malignant tail at 30."""
    head = {"N", "PB", "UDH", "IC"}
    floor = 30
    target_map = step_targets(counts, head, floor)
    labels = sorted(counts, key=lambda c: -counts[c])
    return {
        "method": "Step",
        "labels": labels,
        "native": [counts[c] for c in labels],
        "target": [target_map[c] for c in labels],
        "floor": floor,
    }


def _panda_panel(counts: dict[str, int]) -> dict:
    """PANDA: exponential decay fixed to ordinal ISUP grade order, floor 50."""
    order = ["ISUP0", "ISUP1", "ISUP2", "ISUP3", "ISUP4", "ISUP5"]
    native = [counts[c] for c in order]
    floor = 50
    return {
        "method": "Ordinal exp.\\ decay",
        "labels": order,
        "native": native,
        "target": exp_decay_targets(native, floor),
        "floor": floor,
    }


def _camelyon_panel(counts: dict[str, int]) -> dict:
    """CAMELYON16: fixed 10:1 normal:tumor ratio, floor 20 tumor slides."""
    floor = 20
    target = ratio_downsample(counts["normal"], counts["tumor"], ratio=10, floor=floor)
    return {
        "method": "Ratio downsampling (10:1)",
        "labels": ["normal", "tumor"],
        "native": [counts["normal"], counts["tumor"]],
        "target": [counts["normal"], target],
        "floor": floor,
    }


def build_targets(rows: list[dict]) -> dict[str, dict]:
    """Build the controlled-imbalance construction panel for each dataset."""
    by_dataset = {r["dataset"]: r["slide"]["counts"] for r in rows}
    return {
        "TCGA-UT": _tcga_panel(by_dataset["TCGA-UT"]),
        "BRACS": _bracs_panel(by_dataset["BRACS"]),
        "PANDA": _panda_panel(by_dataset["PANDA"]),
        "CAMELYON16": _camelyon_panel(by_dataset["CAMELYON16"]),
    }


def main() -> None:
    """Generate the controlled-construction figure and the targets table."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    rows = json.loads(COUNTS.read_text(encoding="utf-8"))
    targets = build_targets(rows)
    fig_path = FIGURES_DIR / "controlled_imbalance_construction.png"
    plot_controlled_construction(targets, fig_path)
    write_targets_table(targets, TABLES_DIR / "controlled_imbalance_targets.tex")
    logger.info("Wrote figures to %s and table to %s", FIGURES_DIR, TABLES_DIR)


def _self_check() -> None:
    """Sanity-check the constructions: floors respected, targets never exceed native."""
    native = [792, 400, 100, 28]
    decay = exp_decay_targets(native, 20)
    assert decay[0] == 792, "head class keeps native support"
    assert all(t >= 20 for t in decay), "no target below the floor"
    assert all(t <= n for t, n in zip(decay, native)), "decay only downsamples"

    step = step_targets({"a": 200, "b": 90, "c": 40}, {"a"}, 30)
    assert step == {"a": 200, "b": 30, "c": 30}, "head kept, tail floored"

    assert ratio_downsample(238, 160, ratio=10, floor=20) == 24
    assert ratio_downsample(238, 15, ratio=10, floor=20) == 15, "caps at native min"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _self_check()
    main()
