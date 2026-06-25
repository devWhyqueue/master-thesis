"""Deterministic ROI tiling for BRACS."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

logger = logging.getLogger(__name__)


def tile_rois(
    metadata: pd.DataFrame,
    image_index: dict[str, Path],
    tile_root: Path,
    tile_size: int,
    max_tiles_per_roi: int,
) -> pd.DataFrame:
    """Tile ROI images deterministically and return a row-level patch manifest."""
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for _, row in metadata.iterrows():
        roi_id = str(row["roi_id"])
        source = image_index.get(Path(roi_id).stem)
        if source is None:
            missing.append(roi_id)
            continue
        rows.extend(_tile_one_roi(row, source, tile_root, tile_size, max_tiles_per_roi))
    if missing:
        logger.warning("Missing %d ROI images, examples=%s", len(missing), missing[:10])
    return pd.DataFrame(rows)


def _tile_one_roi(
    row: pd.Series, source: Path, tile_root: Path, tile_size: int, max_tiles: int
) -> list[dict[str, Any]]:
    with Image.open(source) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        coords = [
            (x, y)
            for y in range(0, height - tile_size + 1, tile_size)
            for x in range(0, width - tile_size + 1, tile_size)
        ][:max_tiles]
        if not coords:
            return []
        out_dir = tile_root / str(row["slide_id"]) / str(row["roi_id"])
        out_dir.mkdir(parents=True, exist_ok=True)
        return [
            _tile_row(row, rgb, out_dir, tile_size, index, coord)
            for index, coord in enumerate(coords)
        ]


def _tile_row(
    row: pd.Series,
    rgb: Image.Image,
    out_dir: Path,
    tile_size: int,
    index: int,
    coord: tuple[int, int],
) -> dict[str, Any]:
    x, y = coord
    patch_id = f"{row['roi_id']}__{index:03d}_{x}_{y}"
    out_path = out_dir / f"{patch_id}.jpg"
    if not out_path.exists():
        rgb.crop((x, y, x + tile_size, y + tile_size)).save(out_path, quality=95)
    return {
        "dataset": "bracs",
        "case_id": row["case_id"],
        "slide_id": row["slide_id"],
        "roi_id": row["roi_id"],
        "cancer_type": row["cancer_type"],
        "lesion_type": row["lesion_type"],
        "patch_id": patch_id,
        "image_path": str(out_path),
    }
