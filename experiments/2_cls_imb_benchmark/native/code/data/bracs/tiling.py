"""Deterministic ROI tiling for BRACS with a per-WSI median tile budget."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

logger = logging.getLogger(__name__)


def tile_rois(
    metadata: pd.DataFrame,
    image_index: dict[str, Path],
    tile_root: Path,
    tile_size: int,
) -> tuple[pd.DataFrame, int]:
    """Tile ROI images and cap each WSI at the median available tiles per WSI.

    Returns the row-level patch manifest and the data-driven WSI-bag budget
    (the median of per-WSI available-tile counts), mirroring the median bag
    budget used for CAMELYON16 and PANDA.
    """
    plans = _roi_plans(metadata, image_index, tile_size)
    if not plans:
        return pd.DataFrame(), 0
    bag_size = _median_tiles_per_wsi(plans)
    logger.info("BRACS bag size (median available tiles/WSI) = %d", bag_size)
    rows: list[dict[str, Any]] = []
    for slide_plans in _by_slide(plans):
        rows.extend(_tile_slide(slide_plans, tile_root, tile_size, bag_size))
    return pd.DataFrame(rows), bag_size


def _roi_plans(
    metadata: pd.DataFrame, image_index: dict[str, Path], tile_size: int
) -> list[dict[str, Any]]:
    """Enumerate deterministic tile coordinates per ROI without writing tiles."""
    plans: list[dict[str, Any]] = []
    missing: list[str] = []
    for _, row in metadata.iterrows():
        roi_id = str(row["roi_id"])
        source = image_index.get(Path(roi_id).stem)
        if source is None:
            missing.append(roi_id)
            continue
        coords = _roi_coords(source, tile_size)
        if coords:
            plans.append({"row": row, "source": source, "coords": coords})
    if missing:
        logger.warning("Missing %d ROI images, examples=%s", len(missing), missing[:10])
    return plans


def _roi_coords(source: Path, tile_size: int) -> list[tuple[int, int]]:
    with Image.open(source) as image:
        width, height = image.size
    return [
        (x, y)
        for y in range(0, height - tile_size + 1, tile_size)
        for x in range(0, width - tile_size + 1, tile_size)
    ]


def _median_tiles_per_wsi(plans: list[dict[str, Any]]) -> int:
    totals: dict[str, int] = {}
    for plan in plans:
        slide_id = str(plan["row"]["slide_id"])
        totals[slide_id] = totals.get(slide_id, 0) + len(plan["coords"])
    return int(np.median(list(totals.values())))


def _by_slide(plans: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for plan in plans:
        groups.setdefault(str(plan["row"]["slide_id"]), []).append(plan)
    return list(groups.values())


def _tile_slide(
    slide_plans: list[dict[str, Any]],
    tile_root: Path,
    tile_size: int,
    bag_size: int,
) -> list[dict[str, Any]]:
    """Uniformly subsample a WSI's ROI tiles to the median budget and write them."""
    flat = [
        (plan_index, coord_index, coord)
        for plan_index, plan in enumerate(slide_plans)
        for coord_index, coord in enumerate(plan["coords"])
    ]
    kept = _subsample(flat, bag_size)
    by_plan: dict[int, list[tuple[int, tuple[int, int]]]] = {}
    for plan_index, coord_index, coord in kept:
        by_plan.setdefault(plan_index, []).append((coord_index, coord))
    rows: list[dict[str, Any]] = []
    for plan_index, items in by_plan.items():
        plan = slide_plans[plan_index]
        rows.extend(_write_roi_tiles(plan, items, tile_root, tile_size))
    return rows


def _subsample(
    flat: list[tuple[int, int, tuple[int, int]]], bag_size: int
) -> list[tuple[int, int, tuple[int, int]]]:
    if len(flat) <= bag_size:
        return flat
    keep = np.linspace(0, len(flat) - 1, bag_size).astype(int)
    return [flat[index] for index in keep]


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
