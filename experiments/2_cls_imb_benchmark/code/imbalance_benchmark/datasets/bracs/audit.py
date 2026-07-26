from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from imbalance_benchmark.common import compute_sha256

_TILE_AUDIT_COLUMNS = {
    "slide_id",
    "image_path",
    "magnification",
    "tile_size",
    "x",
    "y",
    "otsu_foreground_fraction",
    "grayscale_std",
    "canny_edge_count",
    "tissue_neighbors",
    "sha256",
}


def _rule_mask(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["magnification"].astype(str).eq("20x")
        & frame["tile_size"].eq(256)
        & frame["x"].mod(256).eq(0)
        & frame["y"].mod(256).eq(0)
        & frame["otsu_foreground_fraction"].ge(0.10)
        & frame["grayscale_std"].ge(8.0)
        & frame["canny_edge_count"].gt(0)
        & frame["tissue_neighbors"].ge(2)
    )


def _worker_count() -> int:
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    return max(1, int(slurm_cpus)) if slurm_cpus else os.cpu_count() or 1


def _hash_mismatch(pair: tuple[str, str]) -> str | None:
    image_path, expected_hash = pair
    path = Path(image_path)
    if not path.is_file() or compute_sha256(path) != expected_hash:
        return image_path
    return None


def validate_tile_manifest(frame: pd.DataFrame, expected_slides: int) -> None:
    """Validate BRACS tile-level evidence for every declared preprocessing rule."""
    missing = _TILE_AUDIT_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"BRACS tile audit fields are missing: {sorted(missing)}")
    if frame["slide_id"].astype(str).nunique() != expected_slides:
        raise ValueError("BRACS tile audit does not cover the full WSI cohort")
    if not bool(_rule_mask(frame).all()):
        raise ValueError(
            "BRACS tile audit contains evidence that violates preprocessing"
        )
    if frame.duplicated(["slide_id", "x", "y"]).any():
        raise ValueError("BRACS tile audit contains duplicate level-0 coordinates")
    pairs = list(
        frame[["image_path", "sha256"]].astype(str).itertuples(index=False, name=None)
    )
    with ProcessPoolExecutor(max_workers=_worker_count()) as pool:
        for mismatch in pool.map(_hash_mismatch, pairs, chunksize=256):
            if mismatch is not None:
                raise ValueError(f"BRACS tile audit hash mismatch: {mismatch}")


def load_tile_manifest(
    manifest_path: Path, tile_root: Path, expected_slides: int
) -> pd.DataFrame:
    """Load the auditable BRACS tile inventory and resolve relative image paths."""
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"BRACS tile audit manifest is missing: {manifest_path}"
        )
    frame = pd.read_csv(manifest_path)
    if "image_path" in frame:
        frame["image_path"] = frame["image_path"].map(
            lambda value: str(
                Path(str(value))
                if Path(str(value)).is_absolute()
                else tile_root / str(value)
            )
        )
    validate_tile_manifest(frame, expected_slides)
    return frame.sort_values(["slide_id", "y", "x"]).reset_index(drop=True)
