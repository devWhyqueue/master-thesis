from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from imbalance_benchmark.common import compute_sha256
from imbalance_benchmark.datasets.panda import cell_label

_SELECTION_COLUMNS = {
    "slide_id",
    "slide_label",
    "provider",
    "has_mask",
    "eligible_tile_count",
    "source_level",
    "tile_size",
    "tissue_fraction_min",
    "tissue_intensity_threshold",
}
_TILE_COLUMNS = {
    "patch_id",
    "patch_label",
    "image_path",
    "sha256",
    "level",
    "tile_size",
    "x",
    "y",
    "tissue_fraction",
    "tissue_intensity_threshold",
}


def validate_selection(
    selection: pd.DataFrame,
    official: pd.DataFrame,
    expected_slides: int,
) -> None:
    """Match selected PANDA IDs and targets to the official released cohort."""
    missing = _SELECTION_COLUMNS - set(selection.columns)
    if missing:
        raise ValueError(f"PANDA selection audit fields are missing: {sorted(missing)}")
    selected_ids = selection["slide_id"].astype(str)
    official_ids = official["slide_id"].astype(str)
    if selected_ids.duplicated().any() or official_ids.duplicated().any():
        raise ValueError("PANDA selection audit contains duplicate slide rows")
    if len(selection) != expected_slides or set(selected_ids) != set(official_ids):
        raise ValueError("PANDA selection and official train.csv slide IDs differ")
    expected = official.set_index("slide_id").loc[selected_ids]
    for column in ("slide_label", "provider", "has_mask"):
        actual_values = selection[column].astype(str).str.lower().to_numpy()
        expected_values = expected[column].astype(str).str.lower().to_numpy()
        if not np.array_equal(actual_values, expected_values):
            raise ValueError(f"PANDA selection differs from official {column}")


def validate_tile_inventory(
    selection: pd.DataFrame,
    inventory: dict[str, pd.DataFrame],
    official: pd.DataFrame,
    expected_slides: int,
) -> None:
    """Validate PANDA tiles, hashes, and provider-specific mask labels."""
    selected_ids = selection["slide_id"].astype(str)
    if selected_ids.duplicated().any():
        raise ValueError("PANDA selection audit contains duplicate slide rows")
    missing = _SELECTION_COLUMNS - set(selection.columns)
    if missing:
        raise ValueError(f"PANDA selection audit fields are missing: {sorted(missing)}")
    if len(selection) != expected_slides or set(inventory) != set(selected_ids):
        raise ValueError("PANDA tile audit does not cover the selected biopsy cohort")
    official_rows = official.set_index("slide_id")
    for _, row in selection.iterrows():
        slide_id = str(row["slide_id"])
        tiles = inventory[slide_id]
        _validate_tile_frame(slide_id, tiles, official_rows.loc[slide_id])
        _validate_selection_row(row, tiles)


def _validate_tile_frame(
    slide_id: str, tiles: pd.DataFrame, official: pd.Series
) -> None:
    missing = _TILE_COLUMNS - set(tiles.columns)
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
    _validate_image_hashes(tiles)
    if bool(official["has_mask"]):
        _validate_mask_labels(tiles, official)


def _validate_image_hashes(tiles: pd.DataFrame) -> None:
    for image_path, expected_hash in tiles[["image_path", "sha256"]].itertuples(
        index=False, name=None
    ):
        path = Path(str(image_path))
        if not path.is_file() or compute_sha256(path) != str(expected_hash):
            raise ValueError(f"PANDA tile audit hash mismatch: {path}")


def _validate_mask_labels(tiles: pd.DataFrame, official: pd.Series) -> None:
    mask_path = Path(str(official["mask_path"]))
    if not mask_path.is_file():
        raise ValueError(f"PANDA official mask is missing: {mask_path}")
    with Image.open(mask_path) as mask:
        columns = ["patch_id", "patch_label", "x", "y", "tile_size"]
        for patch_id, patch_label, x, y, tile_size in tiles[columns].itertuples(
            index=False, name=None
        ):
            box = (x, y, x + tile_size, y + tile_size)
            if box[2] > mask.width or box[3] > mask.height:
                raise ValueError("PANDA tile coordinates exceed the aligned mask")
            cell = np.asarray(mask.crop(box))
            if cell.ndim == 3:
                cell = cell[..., 0]
            if patch_label != cell_label(cell, str(official["provider"])):
                raise ValueError(f"PANDA patch label differs from mask for {patch_id}")


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
