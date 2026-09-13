"""Level-0 PANDA tile audit: fuse the band-eligibility scan with tile comparison."""

from __future__ import annotations

import hashlib
import io
import shutil
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import openslide
import pandas as pd
from PIL import Image

from imbalance_benchmark.common import compute_sha256
from imbalance_benchmark.datasets.panda import cell_label
from imbalance_benchmark.datasets.data.panda.grid import (
    BAND_ROWS,
    TILE_SIZE,
    TISSUE_MIN,
    TISSUE_THRESHOLD,
    _band_rgb,
    _band_starts,
    _dims,
    _eligible_mask,
    _source_crop,
)
from imbalance_benchmark.datasets.data.panda.legacy import (
    Resolved,
    Resolver,
    legacy_resolver,
)
from imbalance_benchmark.datasets.data.panda.tiff import (
    official_tiff_guard,
    source_reader,
)


@dataclass(frozen=True)
class _AuditContext:
    row: pd.Series
    source: Image.Image | openslide.OpenSlide
    mask: Image.Image | openslide.OpenSlide | None
    jpeg_mae_max: float
    jpeg_workers: int


def audit_slide(
    row: pd.Series,
    legacy_dir: Path,
    jpeg_mae_max: float,
    manifest_path: Path | None = None,
    jpeg_workers: int = 1,
    band_rows: int = BAND_ROWS,
) -> pd.DataFrame:
    """Recompute canonical coordinates and labels from official PANDA sources."""
    if jpeg_workers < 1:
        raise ValueError("PANDA JPEG audit workers must be positive")
    resolve, legacy_total = legacy_resolver(legacy_dir, manifest_path)
    with (
        official_tiff_guard(),
        source_reader(str(row["image_path"])) as source,
        _mask_reader(row) as mask,
    ):
        context = _AuditContext(row, source, mask, jpeg_mae_max, jpeg_workers)
        results, matched, total = _audit_bands(context, resolve, band_rows)
    ordered = [results.get(index) for index in range(legacy_total)]
    if matched != total or any(record is None for record in ordered):
        raise ValueError("PANDA eligible tile coordinates are missing or extra")
    return pd.DataFrame(ordered)


def copy_audited_tiles(inventory: pd.DataFrame, target_root: Path) -> pd.DataFrame:
    """Copy verified legacy JPEGs only after the global protocol gate passes."""
    copied = inventory.copy()
    targets = []
    for legacy_path, patch_id, sha256 in copied[
        ["legacy_image_path", "patch_id", "sha256"]
    ].itertuples(index=False, name=None):
        slide_id, tile_id = str(patch_id).split("/", maxsplit=1)
        target = target_root / "tiles" / slide_id / f"{tile_id}.jpg"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(legacy_path), target)
        if compute_sha256(target) != sha256:
            raise ValueError(f"PANDA copied tile differs from audited source: {target}")
        targets.append(str(target))
    copied["image_path"] = targets
    return copied.drop(columns="legacy_image_path")


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


@contextmanager
def _mask_reader(row: pd.Series) -> Iterator[Image.Image | openslide.OpenSlide | None]:
    if not bool(row["has_mask"]):
        yield None
        return
    path = Path(str(row["mask_path"]))
    if not path.is_file():
        raise ValueError(f"PANDA official mask is missing: {path}")
    with source_reader(str(path)) as mask:
        yield mask


def _audit_bands(
    context: _AuditContext, resolve: Resolver, band_rows: int
) -> tuple[dict[int, dict[str, object]], int, int]:
    """Single band pass: compute eligibility and audit its tiles together.

    Returns ``(results_by_position, matched_count, total_eligible_count)``.
    """
    width, height = _dims(context.source)
    results: dict[int, dict[str, object]] = {}
    matched = 0
    k = 0
    with ThreadPoolExecutor(max_workers=context.jpeg_workers) as pool:
        for y, rows in _band_starts(height, band_rows):
            rgb = _band_rgb(
                _source_crop(context.source, 0, y, width, rows), width, rows
            )
            batch = _eligible_batch(rgb, y, resolve, k)
            k += len(batch)
            jpgs = iter(
                pool.map(
                    _legacy_pixels_and_hash,
                    (item[3][2] for item in batch if item[3] is not None),
                )
            )
            for cell_y, x, cell, resolved in batch:
                if resolved is None:
                    continue
                position, patch_id, legacy_path = resolved
                actual, sha256 = next(jpgs)
                results[position] = _audit_tile(
                    context, patch_id, legacy_path, x, cell_y, cell, actual, sha256
                )
                matched += 1
    return results, matched, k


def _eligible_batch(
    rgb: np.ndarray, y: int, resolve: Resolver, k: int
) -> list[tuple[int, int, np.ndarray, Resolved | None]]:
    eligible = _eligible_mask(rgb)
    batch: list[tuple[int, int, np.ndarray, Resolved | None]] = []
    for row_index in range(eligible.shape[0]):
        cell_y = y + row_index * TILE_SIZE
        for col_index in np.flatnonzero(eligible[row_index]):
            x = int(col_index) * TILE_SIZE
            cell = rgb[
                row_index * TILE_SIZE : (row_index + 1) * TILE_SIZE, x : x + TILE_SIZE
            ]
            batch.append((cell_y, x, cell, resolve(k, x, cell_y)))
            k += 1
    return batch


def _audit_tile(
    context: _AuditContext,
    patch_id: str,
    legacy_path: Path,
    x: int,
    y: int,
    expected_u8: np.ndarray,
    actual: np.ndarray,
    sha256: str,
) -> dict[str, object]:
    row, mask = context.row, context.mask
    expected = expected_u8.astype(np.int16)
    if (
        actual.shape != expected.shape
        or float(np.abs(actual - expected).mean()) > context.jpeg_mae_max
    ):
        raise ValueError(f"PANDA tile source-crop mismatch: {legacy_path}")
    cell = np.asarray(_source_crop(mask, x, y)) if mask is not None else None
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
        "patch_id": f"{row.slide_id}/{patch_id}",
        "patch_label": label,
        "legacy_image_path": str(legacy_path),
        "sha256": sha256,
        "x": x,
        "y": y,
        "level": 0,
        "tile_size": TILE_SIZE,
        "tissue_fraction_min": TISSUE_MIN,
        "tissue_intensity_threshold": TISSUE_THRESHOLD,
    }


def _legacy_pixels_and_hash(path: Path) -> tuple[np.ndarray, str]:
    data = path.read_bytes()
    with Image.open(io.BytesIO(data)) as tile:
        pixels = np.asarray(tile.convert("RGB"), dtype=np.int16)
    return pixels, hashlib.sha256(data).hexdigest()
