"""Visualise native slide-level class imbalance against the CV long-tail regime.

Native slide-level class distributions (see ``0_datasets``) are only mildly
imbalanced, far below the imbalance factors long-tailed CV benchmarks
(CIFAR-LT) use to stress-test the mitigation methods this report evaluates.
Generates the Datasets-section motivation figure only; no experiments run.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
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
FIGURES_DIR = Path(__file__).resolve().parent / "outputs" / "figures"

PATCH_COLOR = "#4c78a8"
SLIDE_COLOR = "#c8615a"


def entropy_imbalance(counts: list[float]) -> float:
    """Return 1 - normalised entropy of the class-count distribution."""
    c = np.asarray([v for v in counts if v > 0], dtype=float)
    k = c.size
    if k <= 1:
        return 1.0
    p = c / c.sum()
    return float(1.0 - (-(p * np.log(p)).sum()) / np.log(k))


def cifar_lt_reference(n_classes: int, n_max: int, rho: float) -> list[float]:
    """Cui et al. exponential long-tail decay with no floor, for the CV reference band."""
    mu = rho ** (-1 / (n_classes - 1))
    return [n_max * mu**i for i in range(n_classes)]


def cifar_lt_band() -> tuple[float, float]:
    """Return the (low, high) 1-H_norm band spanned by CIFAR-10-LT/CIFAR-100-LT at rho=100."""
    cifar10_lt = entropy_imbalance(cifar_lt_reference(10, 5000, 100))
    cifar100_lt = entropy_imbalance(cifar_lt_reference(100, 500, 100))
    return min(cifar10_lt, cifar100_lt), max(cifar10_lt, cifar100_lt)


def plot_native_regimes(rows: list[dict], path: Path) -> None:
    """Grouped bar chart of native 1-H_norm (patch vs slide) with a CIFAR-LT reference band."""
    datasets = [r["dataset"] for r in rows]
    patch = [entropy_imbalance(list(r["tile"]["counts"].values())) for r in rows]
    slide = [entropy_imbalance(list(r["slide"]["counts"].values())) for r in rows]

    x = np.arange(len(datasets))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axhspan(
        *cifar_lt_band(),
        color="orange",
        alpha=0.15,
        label=r"CIFAR-LT reference band ($\rho=100$)",
    )
    ax.bar(x - width / 2, patch, width, label="Patch/tile", color=PATCH_COLOR)
    ax.bar(x + width / 2, slide, width, label="Slide", color=SLIDE_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel(r"Imbalance $1-H_{\mathrm{norm}}$")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main() -> None:
    """Generate the native-imbalance-vs-CIFAR-LT-band figure."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    rows = json.loads(COUNTS.read_text(encoding="utf-8"))
    path = FIGURES_DIR / "native_imbalance_regimes.png"
    plot_native_regimes(rows, path)
    logger.info("Wrote %s", path)


def _self_check() -> None:
    """Sanity-check the entropy metric and the CIFAR-LT reference band."""
    assert entropy_imbalance([10, 10, 10, 10]) < 1e-9, "uniform -> 0"
    assert entropy_imbalance([100, 0, 0]) == 1.0, "single class -> 1"
    low, high = cifar_lt_band()
    assert 0.0 < low < high < 1.0, "band is a proper, non-degenerate interval"
    assert low > 0.09, "CIFAR-LT band sits above native slide-level imbalance (<=0.085)"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _self_check()
    main()
