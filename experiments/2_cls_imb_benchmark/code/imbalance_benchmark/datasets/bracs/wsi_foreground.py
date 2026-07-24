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
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import openslide
from PIL import Image
from skimage.feature import canny
from skimage.filters import threshold_otsu

__all__ = [
    "TILE_SIZE",
    "Cell",
    "Level",
    "cell_stats",
    "coarse_tissue_mask",
    "grid_shape",
    "is_candidate",
    "read_tile_region",
    "select_level",
]

TILE_SIZE = 256
TARGET_MAGNIFICATION = 20.0
MIN_OTSU_FOREGROUND = 0.10
MIN_GRAYSCALE_STD = 8.0


@dataclass(frozen=True)
class Level:
    """The pyramid level (and its downsample factors) chosen to reach 20x."""

    downsample: float
    level: int
    level_downsample: float


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


def select_level(slide: openslide.OpenSlide) -> Level:
    """Pick the pyramid level closest to 20x and its downsample factors."""
    downsample = _base_magnification(slide) / TARGET_MAGNIFICATION
    level = slide.get_best_level_for_downsample(downsample)
    return Level(downsample, level, slide.level_downsamples[level])


def grid_shape(slide: openslide.OpenSlide, downsample: float) -> tuple[int, int]:
    """Return the (n_cols, n_rows) of non-overlapping 256x256 tiles at 20x."""
    width0, height0 = slide.level_dimensions[0]
    return int(width0 / downsample) // TILE_SIZE, int(height0 / downsample) // TILE_SIZE


def read_tile_region(
    slide: openslide.OpenSlide, row: int, col: int, level_plan: Level
) -> Image.Image:
    """Read one 20x grid cell and resize it to exactly 256x256."""
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


def cell_stats(
    slide: openslide.OpenSlide, row: int, col: int, level_plan: Level
) -> Cell:
    """Compute the Otsu/std/Canny foreground stats for one grid cell."""
    gray = np.asarray(
        read_tile_region(slide, row, col, level_plan).convert("L"), dtype=np.float64
    )
    try:
        foreground_fraction = float((gray < threshold_otsu(gray)).mean())
    except ValueError:
        foreground_fraction = 0.0  # flat tile (e.g. pure background): no Otsu split
    edge_count = int(canny(gray / 255.0).sum())
    return Cell(foreground_fraction, float(gray.std()), edge_count)


def coarse_tissue_mask(slide: openslide.OpenSlide) -> tuple[np.ndarray | None, float]:
    """Cheap low-res Otsu tissue mask used only to skip full-res candidate work.

    Returns ``(None, 1.0)`` when the slide has no coarser pyramid level, or
    when the coarse image is perfectly flat (Otsu can't split it) -- both
    mean "treat every cell as a candidate", the false-safe default.
    """
    if len(slide.level_downsamples) < 2:
        return None, 1.0
    coarse_level = len(slide.level_downsamples) - 1
    coarse_downsample = slide.level_downsamples[coarse_level]
    coarse = np.asarray(
        slide.read_region(
            (0, 0), coarse_level, slide.level_dimensions[coarse_level]
        ).convert("L"),
        dtype=np.float64,
    )
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


def _base_magnification(slide: openslide.OpenSlide) -> float:
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
