"""Level-0 PANDA tile-grid audit."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from imbalance_benchmark.common import compute_sha256
from imbalance_benchmark.datasets.panda import cell_label

TILE_SIZE, TISSUE_MIN, TISSUE_THRESHOLD = 256, 0.35, 210


@dataclass(frozen=True)
class _AuditContext:
    row: pd.Series
    source: Image.Image
    mask: Image.Image | None
    target_root: Path
    jpeg_mae_max: float


def audit_slide(
    row: pd.Series, legacy_dir: Path, target_root: Path, jpeg_mae_max: float
) -> pd.DataFrame:
    """Recompute one level-0 grid and verify every legacy JPEG crop."""
    with Image.open(str(row["image_path"])) as source:
        coords = eligible_coordinates(source)
        legacy = _legacy_tiles(legacy_dir)
        if set(legacy) != set(range(len(coords))):
            raise ValueError("PANDA eligible tile coordinates are missing or extra")
        mask = _load_mask(row)
        try:
            context = _AuditContext(row, source, mask, target_root, jpeg_mae_max)
            records = [
                _audit_tile(context, legacy[index], coord, index)
                for index, coord in enumerate(coords)
            ]
        finally:
            if mask is not None:
                mask.close()
    return pd.DataFrame(records)


def canary_rows(official: pd.DataFrame, legacy_root: Path) -> pd.DataFrame:
    """Select both providers, all ISUP grades, mask states and count extremes."""
    ranked = official.assign(
        tile_count=official["slide_id"].map(
            lambda slide: len(list((legacy_root / str(slide)).glob("*.jpg")))
        )
    )
    selected = [
        group.iloc[0]
        for _, group in ranked.groupby(["provider", "slide_label"], sort=True)
    ]
    selected.extend(group.iloc[0] for _, group in ranked.groupby("has_mask", sort=True))
    selected.extend(
        [ranked.loc[ranked.tile_count.idxmin()], ranked.loc[ranked.tile_count.idxmax()]]
    )
    return pd.DataFrame(selected).drop_duplicates("slide_id").reset_index(drop=True)


def eligible_coordinates(source: Image.Image) -> list[tuple[int, int]]:
    """Return every complete 256-pixel cell meeting locked tissue rule."""
    coords = []
    for y in range(0, source.height - TILE_SIZE + 1, TILE_SIZE):
        for x in range(0, source.width - TILE_SIZE + 1, TILE_SIZE):
            crop = np.asarray(source.crop((x, y, x + TILE_SIZE, y + TILE_SIZE)))
            rgb = crop[..., :3] if crop.ndim == 3 else np.repeat(crop[..., None], 3, 2)
            if (rgb.mean(axis=2) < TISSUE_THRESHOLD).mean() >= TISSUE_MIN:
                coords.append((x, y))
    return coords


def _legacy_tiles(directory: Path) -> dict[int, Path]:
    paths = {
        int(path.stem): path for path in directory.glob("*.jpg") if path.stem.isdigit()
    }
    if len(paths) != len(list(directory.glob("*.jpg"))):
        raise ValueError(f"PANDA legacy tiles have non-numeric names: {directory}")
    return paths


def _load_mask(row: pd.Series) -> Image.Image | None:
    if not bool(row["has_mask"]):
        return None
    path = Path(str(row["mask_path"]))
    if not path.is_file():
        raise ValueError(f"PANDA official mask is missing: {path}")
    return Image.open(path)


def _audit_tile(
    context: _AuditContext,
    legacy: Path,
    coord: tuple[int, int],
    index: int,
) -> dict[str, object]:
    row, source, mask = context.row, context.source, context.mask
    x, y = coord
    with Image.open(legacy) as tile:
        actual = np.asarray(tile.convert("RGB"), dtype=np.int16)
    expected = np.asarray(
        source.crop((x, y, x + TILE_SIZE, y + TILE_SIZE)).convert("RGB"), dtype=np.int16
    )
    if (
        actual.shape != expected.shape
        or float(np.abs(actual - expected).mean()) > context.jpeg_mae_max
    ):
        raise ValueError(f"PANDA tile source-crop mismatch: {legacy}")
    target = context.target_root / "tiles" / str(row.slide_id) / f"{index}.jpg"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(legacy, target)
    cell = np.asarray(mask.crop((x, y, x + TILE_SIZE, y + TILE_SIZE))) if mask else None
    label = (
        "unlabelled"
        if cell is None
        else cell_label(cell[..., 0] if cell.ndim == 3 else cell, str(row.provider))
    )
    return {
        "slide_id": str(row.slide_id),
        "case_id": str(row.slide_id),
        "slide_label": str(row.slide_label),
        "provider": str(row.provider),
        "has_mask": bool(row.has_mask),
        "patch_id": f"{row.slide_id}/{index}",
        "patch_label": label,
        "image_path": str(target),
        "sha256": compute_sha256(target),
        "x": x,
        "y": y,
        "level": 0,
        "tile_size": TILE_SIZE,
        "tissue_fraction_min": TISSUE_MIN,
        "tissue_intensity_threshold": TISSUE_THRESHOLD,
    }
