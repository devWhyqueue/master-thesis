"""Deterministic ROI tiling for BRACS."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

__all__ = ["tile_rois"]


def tile_rois(
    metadata: pd.DataFrame,
    image_index: dict[str, Path],
    tile_root: Path,
    tile_size: int,
) -> pd.DataFrame:
    """Tile every complete patch from each usable ROI image."""
    plans = _roi_plans(metadata, image_index, tile_size)
    rows: list[dict[str, Any]] = []
    for plan in plans:
        items = list(enumerate(plan["coords"]))
        rows.extend(_write_roi_tiles(plan, items, tile_root, tile_size))
    return pd.DataFrame(rows)


def _roi_plans(
    metadata: pd.DataFrame, image_index: dict[str, Path], tile_size: int
) -> list[dict[str, Any]]:
    plans: list[dict[str, Any]] = []
    for _, row in metadata.iterrows():
        source = image_index.get(Path(str(row["roi_id"])).stem)
        if source is None:
            continue
        coords = _roi_coords(source, tile_size)
        if coords:
            plans.append({"row": row, "source": source, "coords": coords})
    return plans


def _roi_coords(source: Path, tile_size: int) -> list[tuple[int, int]]:
    with Image.open(source) as image:
        width, height = image.size
    return [
        (x, y)
        for y in range(0, height - tile_size + 1, tile_size)
        for x in range(0, width - tile_size + 1, tile_size)
    ]


def _write_roi_tiles(
    plan: dict[str, Any],
    items: list[tuple[int, tuple[int, int]]],
    tile_root: Path,
    tile_size: int,
) -> list[dict[str, Any]]:
    row = plan["row"]
    out_dir = tile_root / str(row["slide_id"]) / str(row["roi_id"])
    out_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(plan["source"]) as image:
        rgb = image.convert("RGB")
        return [
            _tile_row(row, rgb, out_dir, tile_size, coord_index, coord)
            for coord_index, coord in items
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
