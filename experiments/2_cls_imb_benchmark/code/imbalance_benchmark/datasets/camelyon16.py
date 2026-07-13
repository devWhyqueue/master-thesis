"""CAMELYON16 slide metadata, mask-derived patch labels, and slide-disjoint splits.

The pre-tiled 20x patches are named by a flat raster index over the slide's mask
grid at 32 mask-pixels per tile. The convention (reverse-engineered and
validated: zero out-of-bounds, spatially coherent tumor labels, zero tumor on
normal slides) is column-major:

    n_rows = mask_H // CELL ; col, row = divmod(patch_id, n_rows)
    cell   = mask[row*CELL:(row+1)*CELL, col*CELL:(col+1)*CELL]

Mask values: 0 = background, 1 = normal tissue, 2 = tumor.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

LABELS = ("normal", "tumor")
SPLITS = ("train", "validation", "test")
CELL = 32
TUMOR = 2
TUMOR_FRACTION = 0.5

# Tumor slides whose annotations are NOT exhaustive (README): unannotated tumor
# would be mislabelled normal, so they are dropped from the patch regime. They
# remain valid for the WSI-bag regime, which uses only the slide-level label.
NON_EXHAUSTIVE_TUMOR = frozenset(
    f"tumor_{index:03d}"
    for index in (
        10,
        15,
        18,
        20,
        25,
        29,
        33,
        34,
        44,
        46,
        51,
        54,
        55,
        56,
        67,
        79,
        85,
        92,
        95,
        110,
    )
)


def load_slide_labels(data_root: Path) -> dict[str, str]:
    """Return the slide-level tumor/normal label for every slide in reference.csv."""
    frame = pd.read_csv(data_root / "metadata" / "reference.csv")
    labels = {
        str(Path(str(image)).stem): str(kind).strip()
        for image, kind in zip(frame["image"], frame["type"])
    }
    unexpected = set(labels.values()) - set(LABELS)
    if unexpected:
        raise ValueError(f"Unexpected CAMELYON16 slide labels: {sorted(unexpected)}")
    return labels


def list_slide_patches(data_root: Path, slide_id: str) -> list[tuple[int, Path]]:
    """Return (patch_id, image_path) pairs for a slide, sorted by integer patch id."""
    patch_dir = data_root / "patches" / "20x" / slide_id
    pairs = [
        (int(entry.name[:-4]), Path(entry.path))
        for entry in os.scandir(patch_dir)
        if entry.name.endswith(".jpg")
    ]
    return sorted(pairs, key=lambda pair: pair[0])


def slides_with_patches(data_root: Path) -> list[str]:
    """Return the sorted slide ids that have an extracted 20x patch directory."""
    patch_root = data_root / "patches" / "20x"
    return sorted(entry.name for entry in os.scandir(patch_root) if entry.is_dir())


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


def split_cases(slide_frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Return stratified case-level split assignments keyed by case_id.

    ``slide_frame`` has one row per slide with ``case_id`` and ``slide_label``.
    Slides are stratified by their slide-level tumor/normal label so tumor and
    normal slides are spread across train/validation/test in ~70/15/15 shares.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict[str, str]] = []
    for _, group in slide_frame.groupby("slide_label", sort=False):
        rows.extend(_split_group(group, rng))
    return pd.DataFrame(rows)


def assert_slide_disjoint(frame: pd.DataFrame) -> None:
    """Raise if any case appears in more than one split."""
    split_counts = cast(pd.Series, frame.groupby("case_id")["split"].nunique())
    leaking = [str(case) for case, count in split_counts.items() if int(count) > 1]
    if leaking:
        raise ValueError(f"CAMELYON16 slide leakage: {leaking[:5]}")


def _split_group(group: pd.DataFrame, rng: np.random.Generator) -> list[dict[str, str]]:
    cases = group["case_id"].astype(str).to_numpy()
    rng.shuffle(cases)
    n_cases = len(cases)
    n_train = max(1, int(round(n_cases * 0.70)))
    n_val = max(1, int(round(n_cases * 0.15)))
    if n_train + n_val >= n_cases and n_cases > 1:
        n_val = max(0, n_cases - n_train - 1)
    return (
        [{"case_id": case, "split": "train"} for case in cases[:n_train]]
        + [
            {"case_id": case, "split": "validation"}
            for case in cases[n_train : n_train + n_val]
        ]
        + [{"case_id": case, "split": "test"} for case in cases[n_train + n_val :]]
    )
