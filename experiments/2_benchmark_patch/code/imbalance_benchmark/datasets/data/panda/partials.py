"""Signed partial and combined-inventory I/O for the PANDA audit pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pandas as pd

from imbalance_benchmark.common import sign_file, verify_signed_file, write_json

_AUDIT_DTYPES: dict[str, Any] = {
    "slide_id": str,
    "case_id": str,
    "slide_label": str,
    "provider": str,
    "has_mask": bool,
    "patch_id": str,
    "patch_label": str,
    "legacy_image_path": str,
    "sha256": str,
    "x": "int64",
    "y": "int64",
    "level": "int64",
    "tile_size": "int64",
    "tissue_fraction_min": "float64",
    "tissue_intensity_threshold": "float64",
}
_INVENTORY_DTYPES = {**_AUDIT_DTYPES, "shard_index": "int64"}


def _partials_dir(cfg: dict[str, Any]) -> Path:
    return Path(cfg["canonical_inventory_path"]).parent / "partials"


def _audit_partial_paths(
    cfg: dict[str, Any], shard_index: int
) -> tuple[Path, Path, Path]:
    directory = _partials_dir(cfg)
    return (
        directory / f"audit-{shard_index}.csv",
        directory / f"raw-{shard_index}.json",
        directory / f"crashed-{shard_index}.json",
    )


def audit_partial_done(cfg: dict[str, Any], shard_index: int) -> bool:
    """Return whether one audit shard's signed partial set is complete and unaltered."""
    paths = _audit_partial_paths(cfg, shard_index)
    if not all(path.is_file() for path in paths):
        return False
    for path in paths:
        verify_signed_file(path)
    return True


def write_audit_partial(
    cfg: dict[str, Any],
    shard_index: int,
    frame: pd.DataFrame,
    raw_hashes: dict[str, dict[str, str | None]],
    crashed_slide_ids: list[str],
) -> None:
    """Persist one audit shard's signed tile records, raw-source hashes, and any
    slide IDs whose audit worker crashed (native crash, excluded from ``frame``)."""
    tiles_path, raw_path, crashed_path = _audit_partial_paths(cfg, shard_index)
    tiles_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(tiles_path, index=False)
    sign_file(tiles_path)
    write_json(raw_path, raw_hashes)
    sign_file(raw_path)
    write_json(crashed_path, {"crashed_slide_ids": crashed_slide_ids})
    sign_file(crashed_path)


def read_audit_partial(
    cfg: dict[str, Any], shard_index: int
) -> tuple[pd.DataFrame, dict[str, dict[str, str | None]], list[str]]:
    """Load and re-verify one audit shard's signed partial set."""
    tiles_path, raw_path, crashed_path = _audit_partial_paths(cfg, shard_index)
    verify_signed_file(tiles_path)
    verify_signed_file(raw_path)
    verify_signed_file(crashed_path)
    frame = pd.read_csv(tiles_path, dtype=cast(Any, _AUDIT_DTYPES))
    raw_hashes = json.loads(raw_path.read_text(encoding="utf-8"))
    crashed = json.loads(crashed_path.read_text(encoding="utf-8"))["crashed_slide_ids"]
    return frame, raw_hashes, crashed


def combined_paths(cfg: dict[str, Any]) -> tuple[Path, Path]:
    """Return the combine stage's signed audit-inventory and raw-hashes paths."""
    directory = Path(cfg["canonical_inventory_path"]).parent
    return directory / "audit_inventory.csv", directory / "raw_hashes.json"


def write_combined_inventory(
    cfg: dict[str, Any],
    inventory: pd.DataFrame,
    raw_hashes: dict[str, dict[str, str | None]],
) -> None:
    """Persist the combine stage's signed audit inventory and raw-source hashes."""
    inventory_path, raw_path = combined_paths(cfg)
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(inventory_path, index=False)
    sign_file(inventory_path)
    write_json(raw_path, raw_hashes)
    sign_file(raw_path)


def read_combined_inventory(cfg: dict[str, Any]) -> pd.DataFrame:
    """Load and re-verify the combine stage's signed audit inventory."""
    inventory_path, _ = combined_paths(cfg)
    verify_signed_file(inventory_path)
    return pd.read_csv(inventory_path, dtype=cast(Any, _INVENTORY_DTYPES))


def read_raw_hashes(cfg: dict[str, Any]) -> dict[str, dict[str, str | None]]:
    """Load and re-verify the combine stage's signed raw-source hashes."""
    _, raw_path = combined_paths(cfg)
    verify_signed_file(raw_path)
    return cast(
        dict[str, dict[str, str | None]],
        json.loads(raw_path.read_text(encoding="utf-8")),
    )
