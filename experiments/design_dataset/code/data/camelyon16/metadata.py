"""CAMELYON16 slide metadata and patch discovery."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

LABELS = ("normal", "tumor")

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
