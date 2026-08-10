"""Level-0 PANDA tile-grid audit."""

from __future__ import annotations

import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path

import numpy as np
import openslide
import pandas as pd
from PIL import Image

from imbalance_benchmark.common import compute_sha256
from imbalance_benchmark.datasets.panda import cell_label
from imbalance_benchmark.datasets.data.panda_tiff import (
    official_tiff_guard,
    source_reader,
)

TILE_SIZE, TISSUE_MIN, TISSUE_THRESHOLD = 256, 0.35, 210


@dataclass(frozen=True)
class _AuditContext:
    row: pd.Series
    source: Image.Image | openslide.OpenSlide
    mask: Image.Image | None
    jpeg_mae_max: float
    jpeg_workers: int


def audit_slide(
    row: pd.Series,
    legacy_dir: Path,
    jpeg_mae_max: float,
    manifest_path: Path | None = None,
    jpeg_workers: int = 1,
) -> pd.DataFrame:
    """Recompute canonical coordinates and labels from official PANDA sources."""
    if jpeg_workers < 1:
        raise ValueError("PANDA JPEG audit workers must be positive")
    with official_tiff_guard(), source_reader(str(row["image_path"])) as source:
        coords = eligible_coordinates(source)
        legacy = _legacy_records(legacy_dir, coords, manifest_path)
        if {(item["x"], item["y"]) for item in legacy} != set(coords):
            raise ValueError("PANDA eligible tile coordinates are missing or extra")
        mask = _load_mask(row)
        try:
            context = _AuditContext(row, source, mask, jpeg_mae_max, jpeg_workers)
            records = _audit_tiles(context, legacy)
        finally:
            if mask is not None:
                mask.close()
    return pd.DataFrame(records)


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


def eligible_coordinates(
    source: Image.Image | openslide.OpenSlide,
) -> list[tuple[int, int]]:
    """Return every complete tile meeting the locked tissue rule."""
    coords = []
    width, height = (
        source.dimensions if isinstance(source, openslide.OpenSlide) else source.size
    )
    for y in range(0, height - TILE_SIZE + 1, TILE_SIZE):
        stripe = np.asarray(_source_crop(source, 0, y, width))
        for x in range(0, width - TILE_SIZE + 1, TILE_SIZE):
            crop = stripe[:, x : x + TILE_SIZE]
            rgb = crop[..., :3] if crop.ndim == 3 else np.repeat(crop[..., None], 3, 2)
            if (rgb.mean(axis=2) < TISSUE_THRESHOLD).mean() >= TISSUE_MIN:
                coords.append((x, y))
    return coords


def _source_crop(
    source: Image.Image | openslide.OpenSlide, x: int, y: int, width: int = TILE_SIZE
) -> Image.Image:
    if isinstance(source, openslide.OpenSlide):
        return source.read_region((x, y), 0, (width, TILE_SIZE))
    return source.crop((x, y, x + width, y + TILE_SIZE))


def _legacy_records(
    directory: Path, coords: list[tuple[int, int]], manifest_path: Path | None
) -> list[dict[str, object]]:
    if manifest_path is not None:
        return _manifest_records(manifest_path)
    paths = {
        int(path.stem): path for path in directory.glob("*.jpg") if path.stem.isdigit()
    }
    if len(paths) != len(list(directory.glob("*.jpg"))):
        raise ValueError(f"PANDA legacy tiles have non-numeric names: {directory}")
    if set(paths) != set(range(len(coords))):
        raise ValueError("PANDA eligible tile coordinates are missing or extra")
    return [
        {"patch_id": index, "x": x, "y": y, "image_path": path}
        for index, ((x, y), path) in enumerate(
            zip(coords, (paths[i] for i in range(len(coords))), strict=True)
        )
    ]


def _manifest_records(path: Path) -> list[dict[str, object]]:
    required = {"patch_id", "x", "y", "image_path"}
    frame = pd.read_csv(path)
    if (
        required - set(frame)
        or frame.duplicated(["patch_id"]).any()
        or frame.duplicated(["x", "y"]).any()
    ):
        raise ValueError(f"PANDA legacy tile manifest is invalid: {path}")
    return [dict(record) for record in frame.to_dict(orient="records")]


def _load_mask(row: pd.Series) -> Image.Image | None:
    if not bool(row["has_mask"]):
        return None
    path = Path(str(row["mask_path"]))
    if not path.is_file():
        raise ValueError(f"PANDA official mask is missing: {path}")
    return Image.open(path)


def _audit_tiles(
    context: _AuditContext, legacy: list[dict[str, object]]
) -> list[dict[str, object]]:
    width = (
        context.source.dimensions
        if isinstance(context.source, openslide.OpenSlide)
        else context.source.size
    )[0]
    audited: list[dict[str, object] | None] = [None] * len(legacy)
    indexed = sorted(enumerate(legacy), key=lambda item: int(str(item[1]["y"])))
    with ThreadPoolExecutor(max_workers=context.jpeg_workers) as pool:
        for y, items in groupby(indexed, key=lambda item: int(str(item[1]["y"]))):
            stripe = np.asarray(
                _source_crop(context.source, 0, y, width).convert("RGB"),
                dtype=np.int16,
            )
            item_list = list(items)
            jpgs = pool.map(
                _legacy_pixels_and_hash,
                (Path(str(record["image_path"])) for _, record in item_list),
            )
            for (index, record), (actual, sha256) in zip(item_list, jpgs, strict=True):
                x = int(str(record["x"]))
                audited[index] = _audit_tile(
                    context, record, stripe[:, x : x + TILE_SIZE], actual, sha256
                )
    return [record for record in audited if record is not None]


def _audit_tile(
    context: _AuditContext,
    record: dict[str, object],
    expected: np.ndarray,
    actual: np.ndarray,
    sha256: str,
) -> dict[str, object]:
    row, mask = context.row, context.mask
    x, y = int(str(record["x"])), int(str(record["y"]))
    legacy = Path(str(record["image_path"]))
    if (
        actual.shape != expected.shape
        or float(np.abs(actual - expected).mean()) > context.jpeg_mae_max
    ):
        raise ValueError(f"PANDA tile source-crop mismatch: {legacy}")
    patch_id = str(record["patch_id"])
    cell = np.asarray(mask.crop((x, y, x + TILE_SIZE, y + TILE_SIZE))) if mask else None
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
        "legacy_image_path": str(legacy),
        "sha256": sha256,
        "x": x,
        "y": y,
        "level": 0,
        "tile_size": TILE_SIZE,
        "tissue_fraction_min": TISSUE_MIN,
        "tissue_intensity_threshold": TISSUE_THRESHOLD,
    }


def _legacy_pixels_and_hash(path: Path) -> tuple[np.ndarray, str]:
    with Image.open(path) as tile:
        return np.asarray(tile.convert("RGB"), dtype=np.int16), compute_sha256(path)
