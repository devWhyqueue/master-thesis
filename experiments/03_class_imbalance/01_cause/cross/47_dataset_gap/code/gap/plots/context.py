"""Context figures: physical field of view, and authentic example tiles with their neighbours.

Example rows show the confusion pair (query of class A, its nearest training tile of class B, and
the reverse) plus the query of the best-recognised class. Pixels are shown unaltered. Only BRACS
carries a scale bar: its tiles are cut at a verified 0.25 µm/px, whereas a TCGA-UT patch spans
an unrecorded 128-256 µm.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from PIL import Image

from gap.plots.codes import short_name
from gap.plots.style import COLOR, DATASETS, INK, LABEL, PAIRS, bare_axis, save

__all__ = ["scale_figure", "tiles_figure"]

_BRACS_UM_PER_PX = 0.25
_BAR_UM = 25.0
_ADH_LIMIT_UM = 2000
_BOXES = (
    (256, "TCGA-UT patch, 128-256 µm", False),
    (128, None, False),
    (64, "BRACS tile, 64 µm", True),
)


def scale_figure(path: Path) -> None:
    """Physical field of view of one model input against the 2 mm ADH extent threshold."""
    fig, ax = plt.subplots(figsize=(3.0, 3.0))
    limit = Rectangle((0, 0), _ADH_LIMIT_UM, _ADH_LIMIT_UM, fill=False, edgecolor=INK)
    ax.add_patch(limit)
    title = "2 mm: extent limit separating ADH from DCIS"
    ax.text(_ADH_LIMIT_UM / 2, 2040, title, ha="center", fontsize=6.5)
    for size, label, filled in _BOXES:
        color = COLOR["bracs"] if filled else COLOR["tcga_ut"]
        box = Rectangle((100, 100), size, size, fill=filled, color=color, label=label)
        ax.add_patch(box)
    ax.set_xlim(-50, 2100)
    ax.set_ylim(-50, 2150)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.legend(fontsize=6.5, loc="center")
    save(fig, path)


def _cells(examples: pd.DataFrame, best: str, pair: tuple[str, str]) -> list[pd.Series]:
    """Query A, nearest B to it, query B, nearest A to it, best-class query."""
    a, b = pair
    wanted = [
        (a, "query"),
        (a, "partner_nn"),
        (b, "query"),
        (b, "partner_nn"),
        (best, "query"),
    ]
    match = [(examples["class"] == c) & (examples["role"] == r) for c, r in wanted]
    return [examples[m].iloc[0] for m in match]


def _caption(cell: pd.Series) -> str:
    name = short_name(str(cell["cancer_type"]))
    if cell["role"] == "query":
        return f"test: {name}\ncorrect in {float(cell['r1_correct_rate']) * 10:.0f}/10 fits"
    return f"nearest {name}\n(training, d = {float(cell['cosine_distance']):.2f})"


def _scale_bar(ax: Any, img: np.ndarray) -> None:
    """25 µm bar in the lower right corner of a BRACS tile."""
    px = _BAR_UM / _BRACS_UM_PER_PX
    h, w = img.shape[:2]
    ax.plot([w - 12 - px, w - 12], [h - 14, h - 14], color="white", linewidth=2.5)
    label = f"{_BAR_UM:.0f} µm"
    style = {"color": "white", "fontsize": 6, "ha": "center", "va": "bottom"}
    ax.text(w - 12 - px / 2, h - 22, label, **style)


def _tile_row(axes: Any, report_dir: Path, ds: str, analysis: dict[str, Any]) -> None:
    """Five tiles of one dataset with captions (and scale bars on BRACS)."""
    examples = pd.read_csv(report_dir / f"examples_{ds}.csv", dtype={"patch_id": str})
    tiles = report_dir / "tiles" / ds
    files = pd.read_csv(tiles / "provenance.csv", dtype=str)
    file_of = dict(zip(files["patch_id"], files["file"]))
    best = analysis["class_names"][int(np.argmax(np.diag(analysis["confusion_r1"])))]
    for ax, cell in zip(axes, _cells(examples, best, PAIRS[ds])):
        img = np.asarray(Image.open(tiles / file_of[cell["patch_id"]]))
        ax.imshow(img)
        bare_axis(ax)
        ax.set_xlabel(_caption(cell), fontsize=6.5, labelpad=2)
        if ds == "bracs":
            _scale_bar(ax, img)
    axes[0].set_ylabel(LABEL[ds], fontsize=8)


def tiles_figure(report_dir: Path, data: dict[str, dict[str, Any]], path: Path) -> None:
    """Two rows (TCGA-UT, BRACS) of five tiles with role, class, and balanced-fit outcome."""
    fig, axes = plt.subplots(2, 5, figsize=(6.3, 3.3))
    for row, ds in zip(axes, DATASETS):
        _tile_row(row, report_dir, ds, data[ds])
    fig.subplots_adjust(wspace=0.06, hspace=0.45)
    save(fig, path, dpi=300)
