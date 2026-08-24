"""Per-tile foreground stats and pyramid-level selection for BRACS WSI tiling.

Performance note / documented protocol adaptation: BRACS slides scanned at
40x have no native ~2x pyramid level, so every 20x tile is read from
full-resolution level 0 -- for a large slide the naive exhaustive grid is
tens of thousands of tiles, each needing a full-res read plus Canny. A cheap
low-resolution Otsu tissue mask (the slide's coarsest pyramid level) is
computed once per slide and used only to *skip* the expensive per-tile
Otsu/std/Canny computation on grid cells with no coarse foreground signal at
all; audited stats for every *kept* tile are still the exact full-resolution
values, never approximated. Because one coarse pixel spans many 20x pixels,
this can in principle miss a very small, low-contrast, isolated tissue focus
that a fully exhaustive scan would have found -- an accepted, documented
deviation from an exhaustive per-tile scan, not a change to the audit
thresholds themselves.

Pyramid I/O (level/resolution selection, region reads, and objective-power
resolution) is delegated to tiatoolbox's ``WSIReader``, which resolves
20x-power reads and the mpp-based objective-power fallback internally
instead of re-implementing OpenSlide-level math by hand.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image
from skimage.feature import canny
from skimage.filters import threshold_otsu
from tiatoolbox.wsicore.wsireader import WSIReader

__all__ = [
    "TILE_SIZE",
    "Cell",
    "cell_stats",
    "coarse_tissue_mask",
    "grid_shape",
    "is_candidate",
    "read_tile_region",
    "select_downsample",
]

TILE_SIZE = 256
TARGET_MAGNIFICATION = 20.0
MIN_OTSU_FOREGROUND = 0.10
MIN_GRAYSCALE_STD = 8.0


@dataclass(frozen=True)
class Cell:
    """One grid cell's audit stats, before the tissue-neighbour isolation rule."""

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


def select_downsample(reader: WSIReader) -> float:
    """Downsample from the slide's base magnification (or mpp fallback) to 20x."""
    power = reader.info.objective_power
    if power is None:
        raise ValueError("Cannot determine BRACS slide base magnification")
    return float(power) / TARGET_MAGNIFICATION


def grid_shape(reader: WSIReader, downsample: float) -> tuple[int, int]:
    """Return the (n_cols, n_rows) of non-overlapping 256x256 tiles at 20x."""
    width0, height0 = reader.info.slide_dimensions
    return int(width0 / downsample) // TILE_SIZE, int(height0 / downsample) // TILE_SIZE


def read_tile_region(
    reader: WSIReader, row: int, col: int, downsample: float
) -> Image.Image:
    """Read one 20x grid cell, resampled to exactly 256x256 by tiatoolbox."""
    level0_x = int(col * TILE_SIZE * downsample)
    level0_y = int(row * TILE_SIZE * downsample)
    region = reader.read_rect(
        (level0_x, level0_y),
        (TILE_SIZE, TILE_SIZE),
        resolution=TARGET_MAGNIFICATION,
        units="power",
    )
    return Image.fromarray(region, mode="RGB")


def cell_stats(reader: WSIReader, row: int, col: int, downsample: float) -> Cell:
    """Compute the Otsu/std/Canny foreground stats for one grid cell."""
    gray = np.asarray(
        read_tile_region(reader, row, col, downsample).convert("L"), dtype=np.float64
    )
    try:
        foreground_fraction = float((gray < threshold_otsu(gray)).mean())
    except ValueError:
        foreground_fraction = 0.0  # flat tile (e.g. pure background): no Otsu split
    edge_count = int(canny(gray / 255.0).sum())
    return Cell(foreground_fraction, float(gray.std()), edge_count)


def _read_coarse_grayscale(reader: WSIReader, coarse_level: int) -> np.ndarray:
    width, height = reader.info.level_dimensions[coarse_level]
    region = reader.read_rect(
        (0, 0), (int(width), int(height)), resolution=coarse_level, units="level"
    )
    return np.asarray(Image.fromarray(region).convert("L"), dtype=np.float64)


def coarse_tissue_mask(reader: WSIReader) -> tuple[np.ndarray | None, float]:
    """Cheap low-res Otsu tissue mask used only to skip full-res candidate work.

    Returns ``(None, 1.0)`` when the slide has no coarser pyramid level, or
    when the coarse image is perfectly flat (Otsu can't split it) -- both
    mean "treat every cell as a candidate", the false-safe default.
    """
    level_downsamples = reader.info.level_downsamples
    if len(level_downsamples) < 2:
        return None, 1.0
    coarse_level = len(level_downsamples) - 1
    coarse_downsample = level_downsamples[coarse_level]
    coarse = _read_coarse_grayscale(reader, coarse_level)
    try:
        threshold = threshold_otsu(coarse)
    except ValueError:
        return None, coarse_downsample
    return coarse < threshold, coarse_downsample


def is_candidate(
    mask: np.ndarray | None,
    coarse_downsample: float,
    row: int,
    col: int,
    downsample: float,
) -> bool:
    """Whether a grid cell has any coarse foreground signal worth full-res scoring."""
    if mask is None:
        return True
    scale = downsample / coarse_downsample
    y0, x0 = int(row * TILE_SIZE * scale), int(col * TILE_SIZE * scale)
    y1, x1 = (
        max(y0 + 1, int((row + 1) * TILE_SIZE * scale)),
        max(x0 + 1, int((col + 1) * TILE_SIZE * scale)),
    )
    return bool(mask[y0:y1, x0:x1].any())
