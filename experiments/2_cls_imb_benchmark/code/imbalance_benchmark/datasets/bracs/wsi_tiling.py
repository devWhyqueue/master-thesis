"""BRACS whole-slide tiling at 20x into an audited 256x256 tissue grid.

Mirrors the CAMELYON16 foreground stack described in the exp-2 report: a
tile is kept only if it clears the Otsu-foreground/grayscale-std/Canny-edge
primary criteria *and* has at least two 8-neighbours in the grid that also
clear them (isolation removal).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import openslide
import pandas as pd
from PIL import Image
from scipy.ndimage import convolve
from skimage.feature import canny
from skimage.filters import threshold_otsu

from imbalance_benchmark.common import compute_sha256

__all__ = ["discover_slides", "tile_slide"]

TILE_SIZE = 256
TARGET_MAGNIFICATION = 20.0
MIN_OTSU_FOREGROUND = 0.10
MIN_GRAYSCALE_STD = 8.0
MIN_TISSUE_NEIGHBORS = 2
_NEIGHBOR_KERNEL = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])


@dataclass(frozen=True)
class _Level:
    downsample: float
    level: int
    level_downsample: float


@dataclass(frozen=True)
class _Cell:
    otsu_foreground_fraction: float
    grayscale_std: float
    canny_edge_count: int

    @property
    def is_primary(self) -> bool:
        """Whether this tile clears the Otsu/std/Canny foreground thresholds."""
        return (
            self.otsu_foreground_fraction >= MIN_OTSU_FOREGROUND
            and self.grayscale_std >= MIN_GRAYSCALE_STD
            and self.canny_edge_count > 0
        )


def discover_slides(root: Path) -> dict[str, Path]:
    """Return every BRACS_*.svs path under ``root``, keyed by its slide_id."""
    return {path.stem: path for path in sorted(root.rglob("BRACS_*.svs"))}


def tile_slide(svs_path: Path, slide_id: str, tile_root: Path) -> pd.DataFrame:
    """Tile one BRACS WSI at 20x and write the audited, tissue-isolated tiles.

    Returns one row per *kept* tile (Otsu foreground fraction, grayscale std,
    Canny edges, and >=2 tissue neighbours), matching ``bracs/audit.py``.
    """
    with openslide.OpenSlide(str(svs_path)) as slide:
        level_plan = _select_level(slide)
        n_cols, n_rows = _grid_shape(slide, level_plan.downsample)
        if n_cols == 0 or n_rows == 0:
            raise ValueError(f"{slide_id}: slide too small for a single 20x tile")
        grid = _cell_grid(slide, n_rows, n_cols, level_plan)
        keep, neighbor_counts = _keep_mask(grid)
        out_dir = tile_root / slide_id
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = _kept_rows(
            slide, slide_id, out_dir, grid, keep, neighbor_counts, level_plan
        )
    if not rows:
        raise ValueError(f"{slide_id}: no tiles passed the tissue-foreground audit")
    return pd.DataFrame(rows)


def _cell_grid(
    slide: Any, n_rows: int, n_cols: int, level_plan: _Level
) -> list[list[_Cell]]:
    return [
        [_cell_stats(slide, row, col, level_plan) for col in range(n_cols)]
        for row in range(n_rows)
    ]


def _keep_mask(grid: list[list[_Cell]]) -> tuple[np.ndarray, np.ndarray]:
    primary = np.array([[cell.is_primary for cell in row] for row in grid])
    neighbor_counts = convolve(primary.astype(int), _NEIGHBOR_KERNEL, mode="constant")
    return primary & (neighbor_counts >= MIN_TISSUE_NEIGHBORS), neighbor_counts


def _kept_rows(
    slide: Any,
    slide_id: str,
    out_dir: Path,
    grid: list[list[_Cell]],
    keep: np.ndarray,
    neighbor_counts: np.ndarray,
    level_plan: _Level,
) -> list[dict[str, Any]]:
    n_rows, n_cols = keep.shape
    return [
        _write_tile(
            slide,
            slide_id,
            out_dir,
            row,
            col,
            grid[row][col],
            int(neighbor_counts[row, col]),
            level_plan,
        )
        for row in range(n_rows)
        for col in range(n_cols)
        if keep[row, col]
    ]


def _select_level(slide: Any) -> _Level:
    downsample = _base_magnification(slide) / TARGET_MAGNIFICATION
    level = slide.get_best_level_for_downsample(downsample)
    return _Level(downsample, level, slide.level_downsamples[level])


def _base_magnification(slide: Any) -> float:
    # ponytail: scanner metadata is not always the ideal on paper; the MPP
    # fallback is a calibration point if BRACS ever mixes in a scanner whose
    # base power isn't 20x/40x.
    power = slide.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER)
    if power is not None:
        return float(power)
    mpp_x = slide.properties.get(openslide.PROPERTY_NAME_MPP_X)
    if mpp_x is not None:
        return 40.0 if float(mpp_x) < 0.375 else 20.0
    raise ValueError("Cannot determine BRACS slide base magnification")


def _grid_shape(slide: Any, downsample: float) -> tuple[int, int]:
    width0, height0 = slide.level_dimensions[0]
    return int(width0 / downsample) // TILE_SIZE, int(height0 / downsample) // TILE_SIZE


def _read_tile_region(
    slide: Any, row: int, col: int, level_plan: _Level
) -> Image.Image:
    level0_x = int(col * TILE_SIZE * level_plan.downsample)
    level0_y = int(row * TILE_SIZE * level_plan.downsample)
    level_size = max(
        1, round(TILE_SIZE * level_plan.downsample / level_plan.level_downsample)
    )
    region = slide.read_region(
        (level0_x, level0_y), level_plan.level, (level_size, level_size)
    ).convert("RGB")
    if level_size != TILE_SIZE:
        region = region.resize((TILE_SIZE, TILE_SIZE), Image.Resampling.LANCZOS)
    return region


def _cell_stats(slide: Any, row: int, col: int, level_plan: _Level) -> _Cell:
    gray = np.asarray(
        _read_tile_region(slide, row, col, level_plan).convert("L"), dtype=np.float64
    )
    try:
        foreground_fraction = float((gray < threshold_otsu(gray)).mean())
    except ValueError:
        foreground_fraction = 0.0  # flat tile (e.g. pure background): no Otsu split
    edge_count = int(canny(gray / 255.0).sum())
    return _Cell(foreground_fraction, float(gray.std()), edge_count)


def _write_tile(
    slide: Any,
    slide_id: str,
    out_dir: Path,
    row: int,
    col: int,
    cell: _Cell,
    tissue_neighbors: int,
    level_plan: _Level,
) -> dict[str, Any]:
    x, y = col * TILE_SIZE, row * TILE_SIZE
    tile_path = out_dir / f"{slide_id}_{y:06d}_{x:06d}.png"
    if not tile_path.exists():
        _read_tile_region(slide, row, col, level_plan).save(tile_path)
    return {
        "slide_id": slide_id,
        "image_path": str(tile_path),
        "magnification": "20x",
        "tile_size": TILE_SIZE,
        "x": x,
        "y": y,
        "otsu_foreground_fraction": cell.otsu_foreground_fraction,
        "grayscale_std": cell.grayscale_std,
        "canny_edge_count": cell.canny_edge_count,
        "tissue_neighbors": tissue_neighbors,
        "sha256": compute_sha256(tile_path),
    }
