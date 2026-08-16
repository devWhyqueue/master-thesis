"""Level-0 PANDA tile-grid geometry: band reads and the vectorized tissue rule."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import openslide
from PIL import Image

TILE_SIZE, TISSUE_MIN, TISSUE_THRESHOLD = 256, 0.35, 210
BAND_ROWS = 512


def _dims(source: Image.Image | openslide.OpenSlide) -> tuple[int, int]:
    return source.dimensions if isinstance(source, openslide.OpenSlide) else source.size


def _band_starts(height: int, band_rows: int) -> Iterator[tuple[int, int]]:
    """Yield ``(y, rows)`` for every band that contains at least one full tile row.

    ``band_rows`` is a multiple of ``TILE_SIZE`` (the level-0 TIFF tile
    height); the audit path in :mod:`panda_slide_audit` reads each band once
    and fuses the eligibility scan with tile comparison, so no level-0 pixel
    is ever decoded twice.
    """
    y = 0
    while y <= height - TILE_SIZE:
        rows = min(band_rows, height - y)
        rows -= rows % TILE_SIZE
        yield y, rows
        y += band_rows


def _band_rgb(band: Image.Image, width: int, rows: int) -> np.ndarray:
    array = np.asarray(band)
    rgb = (
        array[..., :3]
        if array.ndim == 3 and array.shape[-1] >= 3
        else np.repeat(array[..., None], 3, axis=2)
    )
    return rgb[: rows - rows % TILE_SIZE, : width - width % TILE_SIZE]


def _eligible_mask(rgb: np.ndarray) -> np.ndarray:
    """Vectorized tissue rule: mean(RGB) < threshold <=> sum(RGB) < 3 * threshold."""
    nr, nc = rgb.shape[0] // TILE_SIZE, rgb.shape[1] // TILE_SIZE
    dark = rgb.sum(axis=2, dtype=np.uint16) < 3 * TISSUE_THRESHOLD
    frac = dark.reshape(nr, TILE_SIZE, nc, TILE_SIZE).mean(axis=(1, 3))
    return frac >= TISSUE_MIN


def _source_crop(
    source: Image.Image | openslide.OpenSlide,
    x: int,
    y: int,
    width: int = TILE_SIZE,
    height: int = TILE_SIZE,
) -> Image.Image:
    if isinstance(source, openslide.OpenSlide):
        return source.read_region((x, y), 0, (width, height))
    return source.crop((x, y, x + width, y + height))
