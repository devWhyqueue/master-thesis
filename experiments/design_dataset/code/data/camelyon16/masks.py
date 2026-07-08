"""Per-patch tumor/normal labelling from CAMELYON16 pixel masks.

The pre-tiled 20x patches are named by a flat raster index over the slide's
mask grid at 32 mask-pixels per tile. The convention (reverse-engineered and
validated: zero out-of-bounds, spatially coherent tumor labels, zero tumor on
normal slides) is column-major:

    n_rows = mask_H // CELL ; col, row = divmod(patch_id, n_rows)
    cell   = mask[row*CELL:(row+1)*CELL, col*CELL:(col+1)*CELL]

Mask values: 0 = background, 1 = normal tissue, 2 = tumor.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

CELL = 32
TUMOR = 2
TUMOR_FRACTION = 0.5


def load_mask(data_root: Path, slide_id: str) -> np.ndarray:
    """Load a slide's downsampled tumor/normal mask array."""
    return np.load(data_root / "masks" / f"{slide_id}_mask.npy")


def patch_labels(mask: np.ndarray, patch_ids: list[int]) -> list[str]:
    """Label each patch id tumor/normal from its 32x32 mask cell."""
    n_rows = mask.shape[0] // CELL
    n_cols = mask.shape[1] // CELL
    return [_cell_label(mask, pid, n_rows, n_cols) for pid in patch_ids]


def _cell_label(mask: np.ndarray, patch_id: int, n_rows: int, n_cols: int) -> str:
    col, row = divmod(int(patch_id), n_rows)
    if row >= n_rows or col >= n_cols:
        return "normal"
    cell = mask[row * CELL : (row + 1) * CELL, col * CELL : (col + 1) * CELL]
    if cell.size and float((cell == TUMOR).mean()) >= TUMOR_FRACTION:
        return "tumor"
    return "normal"
