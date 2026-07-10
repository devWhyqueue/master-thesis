"""Tile one PANDA biopsy into 20x tissue tiles (SLURM array task).

PANDA ships raw pyramidal ``.tiff`` biopsies (no pre-tiled patches), so tiling
is done here from scratch with PIL (the only imaging library in the container).
The image is read at level 0 (~20x, Virchow2's native scale); tiles are kept by
an image-based tissue filter and, when a mask is present, labelled cancer/benign
from the level-1 mask. One array task handles one slide and is idempotent.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from data.panda.masks import cell_label, load_mask_channel

# Level-0 PANDA pyramids exceed PIL's decompression-bomb guard; lift it process-wide.
setattr(Image, "MAX_IMAGE_PIXELS", None)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse PANDA tiling arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-path", required=True)
    parser.add_argument("--tile-root", required=True)
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--array-task-id", type=int, default=None)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--image-level", type=int, default=0)
    parser.add_argument("--mask-level", type=int, default=1)
    parser.add_argument("--max-tiles", type=int, default=1000)
    parser.add_argument("--tissue-threshold", type=float, default=0.35)
    parser.add_argument("--tissue-value", type=int, default=210)
    return parser.parse_args()


def main() -> None:
    """Tile the slide at this array index."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    selection = pd.read_csv(args.selection_path)
    task_id = _resolve_task_id(args)
    if task_id >= len(selection):
        logger.info(
            "Task %d exceeds %d slides; nothing to do.", task_id, len(selection)
        )
        return
    row = selection.iloc[task_id]
    out_csv = Path(args.manifest_dir) / f"{row['slide_id']}.csv"
    if out_csv.exists():
        logger.info("Skipping already-tiled slide: %s", row["slide_id"])
        return
    tile_slide(row, args, out_csv)


def tile_slide(row: pd.Series, args: argparse.Namespace, out_csv: Path) -> None:
    """Tile one slide, save tissue tiles as jpg, and write its tile manifest."""
    image = _read_level(str(row["image_path"]), args.image_level)
    coords = _cap(_tissue_tiles(image, args), args.max_tiles)
    labels = _tile_labels(row, coords, args, image.shape[0])
    tile_dir = Path(args.tile_root) / str(row["slide_id"])
    tile_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        _save_tile(image, tile_dir, row, index, coord, args.tile_size, label)
        for index, (coord, label) in enumerate(zip(coords, labels))
    ]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    logger.info("Wrote %d tiles for %s", len(rows), row["slide_id"])


def _read_level(path: str, level: int) -> np.ndarray:
    with Image.open(path) as image:
        image.seek(min(level, image.n_frames - 1))
        return np.asarray(image.convert("RGB"))


def _tissue_tiles(image: np.ndarray, args: argparse.Namespace) -> list[tuple[int, int]]:
    size = args.tile_size
    gray = image.mean(axis=2)
    tissue = gray < args.tissue_value
    height, width = gray.shape
    coords = []
    for y in range(0, height - size + 1, size):
        for x in range(0, width - size + 1, size):
            if (
                float(tissue[y : y + size, x : x + size].mean())
                >= args.tissue_threshold
            ):
                coords.append((x, y))
    return coords


def _cap(coords: list[tuple[int, int]], max_tiles: int) -> list[tuple[int, int]]:
    if len(coords) <= max_tiles:
        return coords
    keep = np.linspace(0, len(coords) - 1, max_tiles).astype(int)
    return [coords[index] for index in keep]


def _tile_labels(
    row: pd.Series,
    coords: list[tuple[int, int]],
    args: argparse.Namespace,
    image_height: int,
) -> list[str]:
    if not bool(row["has_mask"]):
        return ["unknown"] * len(coords)
    mask = load_mask_channel(str(row["mask_path"]), args.mask_level)
    scale = max(1, round(image_height / mask.shape[0]))
    provider = str(row["provider"])
    size = args.tile_size
    return [
        cell_label(
            mask[y // scale : (y + size) // scale, x // scale : (x + size) // scale],
            provider,
        )
        for x, y in coords
    ]


def _save_tile(
    image: np.ndarray,
    tile_dir: Path,
    row: pd.Series,
    index: int,
    coord: tuple[int, int],
    size: int,
    label: str,
) -> dict[str, object]:
    x, y = coord
    out_path = tile_dir / f"{index}.jpg"
    if not out_path.exists():
        Image.fromarray(image[y : y + size, x : x + size]).save(out_path, quality=95)
    return {
        "dataset": "panda",
        "slide_id": row["slide_id"],
        "patch_id": index,
        "x": x,
        "y": y,
        "image_path": str(out_path),
        "patch_label": label,
    }


def _resolve_task_id(args: argparse.Namespace) -> int:
    if args.array_task_id is not None:
        return args.array_task_id
    env_value = os.environ.get("SLURM_ARRAY_TASK_ID")
    if env_value is None:
        raise ValueError("Pass --array-task-id or run in a SLURM array.")
    return int(env_value)


if __name__ == "__main__":
    main()
