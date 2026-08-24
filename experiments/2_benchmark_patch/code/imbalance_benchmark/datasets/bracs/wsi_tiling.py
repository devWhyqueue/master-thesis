"""BRACS whole-slide tiling at 20x into an audited 256x256 tissue grid.

Mirrors the CAMELYON16 foreground stack described in the exp-2 report: a
tile is kept only if it clears the Otsu-foreground/grayscale-std/Canny-edge
primary criteria *and* has at least two 8-neighbours in the grid that also
clear them (isolation removal). Per-tile foreground stats, pyramid-level
selection, and the coarse tissue-mask prescreen live in ``wsi_foreground``.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
from scipy.ndimage import convolve
from tiatoolbox.wsicore.wsireader import WSIReader

from imbalance_benchmark.common import compute_sha256
from imbalance_benchmark.datasets.bracs import wsi_foreground as fg

__all__ = ["discover_slides", "tile_slide"]

MIN_TISSUE_NEIGHBORS = 2
_NEIGHBOR_KERNEL = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])


def discover_slides(root: Path) -> dict[str, Path]:
    """Return every BRACS_*.svs path under ``root``, keyed by its slide_id."""
    return {path.stem: path for path in sorted(root.rglob("BRACS_*.svs"))}


@contextmanager
def _open_reader(svs_path: Path) -> Iterator[WSIReader]:
    # tiatoolbox's WSIReader has no context-manager/close protocol of its own;
    # close the underlying OpenSlide handle explicitly so batch runs over
    # hundreds of slides don't accumulate open file descriptors.
    reader = WSIReader.open(str(svs_path))
    try:
        yield reader
    finally:
        openslide_wsi = getattr(reader, "openslide_wsi", None)
        if openslide_wsi is not None:
            openslide_wsi.close()


def tile_slide(svs_path: Path, slide_id: str, tile_root: Path) -> pd.DataFrame:
    """Tile one BRACS WSI at 20x and write the audited, tissue-isolated tiles.

    Returns one row per *kept* tile (Otsu foreground fraction, grayscale std,
    Canny edges, and >=2 tissue neighbours), matching ``bracs/audit.py``.
    """
    with _open_reader(svs_path) as reader:
        downsample = fg.select_downsample(reader)
        n_cols, n_rows = fg.grid_shape(reader, downsample)
        if n_cols == 0 or n_rows == 0:
            raise ValueError(f"{slide_id}: slide too small for a single 20x tile")
        coarse_mask, coarse_downsample = fg.coarse_tissue_mask(reader)
        grid = _cell_grid(
            reader, n_rows, n_cols, downsample, coarse_mask, coarse_downsample
        )
        keep, neighbor_counts = _keep_mask(grid)
        out_dir = tile_root / slide_id
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = _kept_rows(
            reader, slide_id, out_dir, grid, keep, neighbor_counts, downsample
        )
    if not rows:
        raise ValueError(f"{slide_id}: no tiles passed the tissue-foreground audit")
    return pd.DataFrame(rows)


def _cell_grid(
    reader: WSIReader,
    n_rows: int,
    n_cols: int,
    downsample: float,
    coarse_mask: np.ndarray | None,
    coarse_downsample: float,
) -> list[list[fg.Cell]]:
    return [
        [
            fg.cell_stats(reader, row, col, downsample)
            if fg.is_candidate(coarse_mask, coarse_downsample, row, col, downsample)
            else fg.Cell(0.0, 0.0, 0)
            for col in range(n_cols)
        ]
        for row in range(n_rows)
    ]


def _keep_mask(grid: list[list[fg.Cell]]) -> tuple[np.ndarray, np.ndarray]:
    primary = np.array([[cell.is_primary for cell in row] for row in grid])
    neighbor_counts = convolve(primary.astype(int), _NEIGHBOR_KERNEL, mode="constant")
    return primary & (neighbor_counts >= MIN_TISSUE_NEIGHBORS), neighbor_counts


def _kept_rows(
    reader: WSIReader,
    slide_id: str,
    out_dir: Path,
    grid: list[list[fg.Cell]],
    keep: np.ndarray,
    neighbor_counts: np.ndarray,
    downsample: float,
) -> list[dict[str, Any]]:
    n_rows, n_cols = keep.shape
    return [
        _write_tile(
            reader,
            slide_id,
            out_dir,
            row,
            col,
            grid[row][col],
            int(neighbor_counts[row, col]),
            downsample,
        )
        for row in range(n_rows)
        for col in range(n_cols)
        if keep[row, col]
    ]


def _write_tile(
    reader: WSIReader,
    slide_id: str,
    out_dir: Path,
    row: int,
    col: int,
    cell: fg.Cell,
    tissue_neighbors: int,
    downsample: float,
) -> dict[str, Any]:
    x, y = col * fg.TILE_SIZE, row * fg.TILE_SIZE
    tile_path = out_dir / f"{slide_id}_{y:06d}_{x:06d}.png"
    if not tile_path.exists():
        fg.read_tile_region(reader, row, col, downsample).save(tile_path)
    return {
        "slide_id": slide_id,
        "image_path": str(tile_path),
        "magnification": "20x",
        "tile_size": fg.TILE_SIZE,
        "x": x,
        "y": y,
        "otsu_foreground_fraction": cell.otsu_foreground_fraction,
        "grayscale_std": cell.grayscale_std,
        "canny_edge_count": cell.canny_edge_count,
        "tissue_neighbors": tissue_neighbors,
        "sha256": compute_sha256(tile_path),
    }
