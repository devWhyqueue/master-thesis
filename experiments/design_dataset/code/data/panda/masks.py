"""Provider-consistent binary cancer/benign patch labelling from PANDA masks.

PANDA masks are stored in channel 0 of a pyramidal RGB TIFF sharing the image
geometry. The value semantics are provider-specific (verified on-cluster):

    Radboud:    0 background, 1 stroma, 2 benign epithelium, 3/4/5 Gleason 3/4/5
    Karolinska: 0 background, 1 benign, 2 cancer

Both collapse to a common binary tile label: a tile is ``cancer`` when at least
``CANCER_FRACTION`` of its mask cell carries a cancer value, else ``benign``.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

CANCER_VALUES = {"radboud": (3, 4, 5), "karolinska": (2,)}
CANCER_FRACTION = 0.5


def load_mask_channel(mask_path: str, level: int) -> np.ndarray:
    """Return the channel-0 mask value array at the given pyramid level."""
    with Image.open(mask_path) as mask:
        mask.seek(min(level, mask.n_frames - 1))
        array = np.asarray(mask)
    return array[..., 0] if array.ndim == 3 else array


def cell_label(cell: np.ndarray, provider: str) -> str:
    """Label a mask cell ``cancer``/``benign`` by its cancer-value fraction."""
    if cell.size == 0:
        return "benign"
    cancer = np.isin(cell, CANCER_VALUES[provider])
    return "cancer" if float(cancer.mean()) >= CANCER_FRACTION else "benign"
