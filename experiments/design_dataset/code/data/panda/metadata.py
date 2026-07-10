"""PANDA slide metadata, label maps, and a stratified subset selector.

The WSI-bag regime grades each biopsy by its 6-class ISUP label (ISUP0..ISUP5);
the patch regime uses a provider-consistent binary cancer/benign label decoded
from the mask (see ``masks.py``). Slides without a mask keep their ISUP label
for the WSI regime but are excluded from the patch regime.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

WSI_LABELS = ("ISUP0", "ISUP1", "ISUP2", "ISUP3", "ISUP4", "ISUP5")
PATCH_LABELS = ("benign", "cancer")


def isup_label(grade: int) -> str:
    """Return the 6-class WSI label for an ISUP grade 0..5."""
    return f"ISUP{int(grade)}"


def load_slide_frame(data_root: Path) -> pd.DataFrame:
    """Return one row per PANDA slide with labels, provider, and mask presence."""
    frame = pd.read_csv(data_root / "train.csv").rename(
        columns={"image_id": "slide_id", "data_provider": "provider"}
    )
    frame["slide_label"] = frame["isup_grade"].map(isup_label)
    unexpected = set(frame["slide_label"]) - set(WSI_LABELS)
    if unexpected:
        raise ValueError(f"Unexpected PANDA ISUP labels: {sorted(unexpected)}")
    images, masks = data_root / "train_images", data_root / "train_label_masks"
    frame["image_path"] = frame["slide_id"].map(lambda s: str(images / f"{s}.tiff"))
    frame["mask_path"] = frame["slide_id"].map(lambda s: str(masks / f"{s}_mask.tiff"))
    frame["has_mask"] = frame["mask_path"].map(lambda p: Path(p).is_file())
    return frame


def select_subset(frame: pd.DataFrame, n_slides: int, seed: int) -> pd.DataFrame:
    """Return a subset of ~``n_slides`` slides stratified by (ISUP grade x provider).

    Sampling preserves each (grade, provider) cell's native share; ``n_slides``
    at or above the cohort size returns the whole frame. Sorted by ``slide_id``
    for deterministic, reproducible ordering.
    """
    if n_slides >= len(frame):
        return frame.sort_values("slide_id").reset_index(drop=True)
    rng = np.random.default_rng(seed)
    fraction = n_slides / len(frame)
    parts = []
    for _, group in frame.groupby(["isup_grade", "provider"], sort=True):
        take = max(1, int(round(len(group) * fraction)))
        take = min(take, len(group))
        chosen = rng.choice(len(group), size=take, replace=False)
        parts.append(group.iloc[np.sort(chosen)])
    subset = pd.concat(parts, ignore_index=True)
    return subset.sort_values("slide_id").reset_index(drop=True)
