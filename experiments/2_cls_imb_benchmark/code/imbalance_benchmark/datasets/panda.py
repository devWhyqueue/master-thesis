"""PANDA slide metadata, mask-derived patch labels, and slide-disjoint splits.

The WSI-bag regime grades each biopsy by its 6-class ISUP label (ISUP0..ISUP5);
the patch regime uses a provider-consistent binary cancer/benign label decoded
from the mask. Slides without a mask keep their ISUP label for the WSI regime
but are excluded from the patch regime.

PANDA masks are stored in channel 0 of a pyramidal RGB TIFF sharing the image
geometry. The value semantics are provider-specific:

    Radboud:    0 background, 1 stroma, 2 benign epithelium, 3/4/5 Gleason 3/4/5
    Karolinska: 0 background, 1 benign, 2 cancer

Both collapse to a common binary tile label: a tile is ``cancer`` when at least
``CANCER_FRACTION`` of its mask cell carries a cancer value, else ``benign``.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from PIL import Image

WSI_LABELS = ("ISUP0", "ISUP1", "ISUP2", "ISUP3", "ISUP4", "ISUP5")
PATCH_LABELS = ("benign", "cancer")
SPLITS = ("train", "validation", "test")
CANCER_VALUES = {"radboud": (3, 4, 5), "karolinska": (2,)}
CANCER_FRACTION = 0.5
__all__ = [
    "CANCER_FRACTION",
    "CANCER_VALUES",
    "PATCH_LABELS",
    "SPLITS",
    "WSI_LABELS",
    "assert_slide_disjoint",
    "cell_label",
    "isup_label",
    "load_mask_channel",
    "load_slide_frame",
    "load_tile_inventory",
    "select_subset",
    "split_cases",
    "validate_tile_inventory",
]
_SELECTION_AUDIT_COLUMNS = {
    "slide_id",
    "eligible_tile_count",
    "source_level",
    "tile_size",
    "tissue_fraction_min",
    "tissue_intensity_threshold",
}
_TILE_AUDIT_COLUMNS = {
    "patch_id",
    "image_path",
    "level",
    "tile_size",
    "x",
    "y",
    "tissue_fraction",
    "tissue_intensity_threshold",
}


def load_tile_inventory(
    selection: pd.DataFrame, tiles_dir: Path
) -> dict[str, pd.DataFrame]:
    """Load every selected PANDA slide's tile CSV without silently dropping slides."""
    inventory = {}
    for slide_id in selection["slide_id"].astype(str):
        path = tiles_dir / f"{slide_id}.csv"
        if not path.is_file():
            raise FileNotFoundError(
                f"PANDA tile audit is missing for {slide_id}: {path}"
            )
        tiles = pd.read_csv(path)
        if "image_path" in tiles:
            tiles["image_path"] = tiles["image_path"].map(
                lambda value: str(
                    Path(str(value))
                    if Path(str(value)).is_absolute()
                    else tiles_dir / str(value)
                )
            )
        inventory[slide_id] = tiles
    return inventory


def validate_tile_inventory(
    selection: pd.DataFrame,
    inventory: dict[str, pd.DataFrame],
    expected_slides: int,
) -> None:
    """Validate the full PANDA level-0, uncapped, 35%-tissue realization."""
    missing = _SELECTION_AUDIT_COLUMNS - set(selection.columns)
    if missing:
        raise ValueError(f"PANDA selection audit fields are missing: {sorted(missing)}")
    if selection["slide_id"].astype(str).nunique() != expected_slides:
        raise ValueError("PANDA selection audit does not cover the full biopsy cohort")
    if set(inventory) != set(selection["slide_id"].astype(str)):
        raise ValueError("PANDA tile audit and selected-slide inventory differ")
    for _, row in selection.iterrows():
        slide_id = str(row["slide_id"])
        tiles = inventory[slide_id]
        _validate_tile_frame(slide_id, tiles)
        _validate_selection_row(row, tiles)


def _validate_tile_frame(slide_id: str, tiles: pd.DataFrame) -> None:
    missing = _TILE_AUDIT_COLUMNS - set(tiles.columns)
    if missing:
        raise ValueError(f"PANDA tile audit fields are missing: {sorted(missing)}")
    valid = (
        tiles["level"].eq(0)
        & tiles["tile_size"].eq(256)
        & tiles["x"].mod(256).eq(0)
        & tiles["y"].mod(256).eq(0)
        & tiles["tissue_fraction"].ge(0.35)
        & tiles["tissue_intensity_threshold"].eq(210)
    )
    if not bool(valid.all()) or tiles.duplicated(["x", "y"]).any():
        raise ValueError(f"PANDA tile audit violates preprocessing for {slide_id}")
    missing_files = [
        str(path) for path in tiles["image_path"] if not Path(str(path)).is_file()
    ]
    if missing_files:
        raise ValueError(
            f"PANDA tile audit references missing images: {missing_files[:3]}"
        )


def _validate_selection_row(row: pd.Series, tiles: pd.DataFrame) -> None:
    declared = (
        int(row["source_level"]) == 0
        and int(row["tile_size"]) == 256
        and np.isclose(float(row["tissue_fraction_min"]), 0.35)
        and int(row["tissue_intensity_threshold"]) == 210
        and int(row["eligible_tile_count"]) == len(tiles)
        and not tiles.empty
    )
    if not declared:
        raise ValueError(f"PANDA selection audit is inconsistent for {row['slide_id']}")


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
        take = min(max(1, int(round(len(group) * fraction))), len(group))
        chosen = rng.choice(len(group), size=take, replace=False)
        parts.append(group.iloc[np.sort(chosen)])
    subset = pd.concat(parts, ignore_index=True)
    return subset.sort_values("slide_id").reset_index(drop=True)


def load_mask_channel(mask_path: str, level: int) -> np.ndarray:
    """Return the channel-0 mask value array at the given pyramid level."""
    with Image.open(mask_path) as mask:
        n_frames = getattr(mask, "n_frames", 1)
        mask.seek(min(level, n_frames - 1))
        array = np.asarray(mask)
    return array[..., 0] if array.ndim == 3 else array


def cell_label(cell: np.ndarray, provider: str) -> str:
    """Label a mask cell ``cancer``/``benign`` by its cancer-value fraction."""
    if cell.size == 0:
        return "benign"
    cancer = np.isin(cell, CANCER_VALUES[provider])
    return "cancer" if float(cancer.mean()) >= CANCER_FRACTION else "benign"


def split_cases(slide_frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Return stratified case-level split assignments keyed by case_id.

    ``slide_frame`` has one row per slide with ``case_id`` and ``slide_label``.
    Each PANDA biopsy is its own case, so slide-disjoint splits are patient-disjoint.
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
        raise ValueError(f"PANDA slide leakage: {leaking[:5]}")


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
