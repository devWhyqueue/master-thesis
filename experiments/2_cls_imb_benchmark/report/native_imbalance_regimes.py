"""Visualise native class imbalance against the long-tail stress-test regime.

Native slide-level class distributions (see ``0_datasets``) are only mildly
imbalanced, far below the imbalance factors the long-tailed-recognition methods
this report evaluates were designed for: Cui et al. (CVPR 2019) construct
long-tailed benchmarks at imbalance factors rho = N_max / N_min up to 200, and
Buda et al. (2018) study step and linear imbalance at comparable ratios. The
figure compares each dataset's native head-to-tail ratio rho against that
rho in [100, 200] band. Generates the Datasets-section motivation figure only;
no experiments run.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from compute_imbalance import head_tail_ratio

logger = logging.getLogger(__name__)

COUNTS = Path(__file__).resolve().parent / "outputs" / "counts" / "counts.json"
FIGURES_DIR = Path(__file__).resolve().parent / "outputs" / "figures"

PATCH_COLOR = "#4c78a8"
SLIDE_COLOR = "#c8615a"

# Long-tail stress-test regime the evaluated methods were designed for
# (Cui et al. 2019 up to rho=200; Buda et al. 2018 comparable ratios).
STRESS_LOW, STRESS_HIGH = 100.0, 200.0


def plot_native_regimes(rows: list[dict], path: Path) -> None:
    """Grouped bar chart of native head-to-tail ratio (patch vs slide)."""
    datasets = [r["dataset"] for r in rows]
    patch = [head_tail_ratio(list(r["tile"]["counts"].values())) for r in rows]
    slide = [head_tail_ratio(list(r["slide"]["counts"].values())) for r in rows]

    x = np.arange(len(datasets))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    band_label = r"Long-tail stress-test regime ($\rho=100$--$200$)"
    ax.axhspan(STRESS_LOW, STRESS_HIGH, color="orange", alpha=0.18, label=band_label)
    ax.bar(x - width / 2, patch, width, label="Patch/tile", color=PATCH_COLOR)
    ax.bar(x + width / 2, slide, width, label="Slide", color=SLIDE_COLOR)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel(r"Head-to-tail ratio $\rho=N_{\max}/N_{\min}$")
    ax.set_ylim(1, 300)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main() -> None:
    """Generate the native-imbalance-vs-stress-test-regime figure."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    rows = json.loads(COUNTS.read_text(encoding="utf-8"))
    path = FIGURES_DIR / "native_imbalance_regimes.png"
    plot_native_regimes(rows, path)
    logger.info("Wrote %s", path)


def _self_check() -> None:
    """Sanity-check the head-to-tail ratio against the stress-test regime."""
    assert abs(head_tail_ratio([10, 10, 10, 10]) - 1.0) < 1e-9, "uniform -> 1"
    assert head_tail_ratio([90, 10]) == 9.0, "9:1 ratio"
    rows = json.loads(COUNTS.read_text(encoding="utf-8"))
    slide = [head_tail_ratio(list(r["slide"]["counts"].values())) for r in rows]
    assert max(slide) < STRESS_LOW, "native slide ratios sit below the stress-test band"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _self_check()
    main()
