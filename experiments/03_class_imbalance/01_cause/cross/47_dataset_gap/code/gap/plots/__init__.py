"""Report figures and quoted numbers, built only from the exported ``report/`` summaries."""

from __future__ import annotations

from pathlib import Path

from gap.plots.context import scale_figure, tiles_figure
from gap.plots.decisions import confusion_figure, margin_figure
from gap.plots.geometry import geometry_figure, pca_figure
from gap.plots.imbalance import imbalance_figure
from gap.plots.style import load_analyses
from gap.plots.summary import write_summary

__all__ = ["build_report_assets"]


def build_report_assets(report_dir: Path) -> None:
    """Write every figure PDF to ``report_dir/figures`` and the numbers to summary.json."""
    data = load_analyses(report_dir)
    out = report_dir / "figures"
    out.mkdir(exist_ok=True)
    geometry_figure(data, out / "geometry.pdf")
    pca_figure(data, out / "pca.pdf")
    confusion_figure(data["bracs"], out / "confusion_bracs.pdf", annotate=True)
    confusion_figure(data["tcga_ut"], out / "confusion_tcga_ut.pdf", annotate=False)
    margin_figure(data, out / "margins.pdf")
    imbalance_figure(data, out / "imbalance.pdf")
    scale_figure(out / "scale.pdf")
    tiles_figure(report_dir, data, out / "tiles.pdf")
    write_summary(report_dir)
