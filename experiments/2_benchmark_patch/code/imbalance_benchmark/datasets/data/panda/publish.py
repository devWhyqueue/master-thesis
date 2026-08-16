"""Immutable PANDA shard and canonical-inventory publication."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, cast

import pandas as pd

from imbalance_benchmark.common import (
    compute_data_hash,
    compute_sha256,
    sign_file,
    verify_signed_file,
    write_json,
)
from imbalance_benchmark.datasets.data.panda.slide_audit import copy_audited_tiles

LOCKED_SLIDES, LOCKED_LABELLED_SLIDES = 10_616, 10_510
LOCKED_BENIGN_PATCHES, LOCKED_CANCER_PATCHES = 3_701_262, 802_902

# These 4 slides segfault OpenSlide/libtiff/libjpeg deterministically, at
# multiple band-read sizes, in a native crash that Python cannot catch. The
# audit stage isolates and excludes them instead of losing a whole shard's
# work; the combine stage accepts them as a known exclusion rather than
# failing on an unrecognized crash. The slide count above still counts the
# true official cohort, since these slides do genuinely exist upstream; the
# labelled and per-class patch counts already reflect their absence.
EXCLUDED_SLIDE_IDS = frozenset(
    {
        "00e6511435645e50673991768a713c66",
        "1e23449104568325e9c5a032351dfdc6",
        "5930e03671314482e9aedb6050d1776d",
        "8e8067699657d35ca314a76b3892307b",
    }
)


_REQUIRED_FIELDS = {
    "raw_root",
    "legacy_tiles_dir",
    "legacy_manifest_dir",
    "scratch_root",
    "shard_root",
    "shard_mount_root",
    "canonical_inventory_path",
    "sidecar_path",
}
_DEFAULT_FIELDS = {
    "shard_count": 48,
    "jpeg_mae_max": 5.0,
    "audit_shard_count": 32,
    "audit_workers": None,
    "audit_io_workers": 2,
    "audit_band_rows": 512,
}


def materialize_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate required project-owned PANDA materialization paths."""
    cfg = config.get("materialize_panda")
    if not isinstance(cfg, dict):
        raise ValueError(
            f"PANDA materialization fields are missing: {sorted(_REQUIRED_FIELDS)}"
        )
    missing = sorted(_REQUIRED_FIELDS - set(cfg))
    if missing:
        raise ValueError(f"PANDA materialization fields are missing: {missing}")
    return {**_DEFAULT_FIELDS, **cfg}


def assert_locked_counts(official: pd.DataFrame, inventory: pd.DataFrame) -> None:
    """Fail unless immutable protocol cohort totals are reproduced exactly."""
    labelled = inventory[inventory.patch_label.isin(("benign", "cancer"))]
    counts = labelled.patch_label.value_counts()
    actual = (
        len(official),
        labelled.slide_id.nunique(),
        int(counts.get("benign", 0)),
        int(counts.get("cancer", 0)),
    )
    expected = (
        LOCKED_SLIDES,
        LOCKED_LABELLED_SLIDES,
        LOCKED_BENIGN_PATCHES,
        LOCKED_CANCER_PATCHES,
    )
    if actual != expected:
        raise ValueError(f"PANDA locked realization differs: {actual} != {expected}")


def balanced_shards(inventory: pd.DataFrame, shard_count: int) -> pd.Series:
    """Assign complete slides greedily to equalize tile counts across 48 shards."""
    if shard_count != 48:
        raise ValueError("PANDA protocol requires exactly 48 shards")
    loads, assignment = [0] * shard_count, {}
    counts = cast(pd.Series, inventory.groupby("slide_id", sort=True).size())
    for slide_id, count in counts.sort_values(ascending=False).items():
        index = min(range(shard_count), key=lambda value: (loads[value], value))
        assignment[str(slide_id)] = index
        loads[index] += int(count)
    return inventory.slide_id.astype(str).map(assignment).astype(int)


def pack_one_shard(
    shard: pd.DataFrame, shard_index: int, scratch_root: Path, cfg: dict[str, Any]
) -> None:
    """Copy one shard's verified JPEGs straight into its tree, then pack it.

    Copies directly from each tile's audited ``legacy_image_path`` into this
    shard's own scratch tree (no intermediate shared-tiles staging), so
    concurrent pack tasks for different shards never share scratch state.
    """
    root = scratch_root / f"shard={shard_index}"
    copied = copy_audited_tiles(shard, root)
    manifest = root / "manifest.csv"
    copied.assign(
        image_path=copied.apply(lambda row: mounted_path(cfg, row), axis=1)
    ).to_csv(manifest, index=False)
    destination = Path(cfg["shard_root"]) / f"shard={shard_index}.sqfs"
    if pack_shard(root, destination, str(cfg.get("squash_command", "squash-dataset"))):
        write_shard_sidecar(manifest, destination)
    else:
        verify_existing_shard(manifest, destination)
    shutil.rmtree(root)


def pack_shard(root: Path, destination: Path, command: str) -> bool:
    """Pack one shard atomically, returning false only for an existing image."""
    partial = destination.with_suffix(destination.suffix + ".partial")
    if destination.is_file():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    if partial.exists():
        partial.unlink()
    subprocess.run([command, str(root), str(partial)], check=True)
    os.replace(partial, destination)
    return True


def write_shard_sidecar(manifest: Path, shard: Path) -> None:
    """Write signed manifest and content digests beside one published shard."""
    sidecar = shard.with_suffix(".manifest.json")
    write_json(
        sidecar,
        {
            "manifest_sha256": compute_sha256(manifest),
            "shard_path": str(shard),
            "shard_sha256": compute_sha256(shard),
        },
    )
    sign_file(sidecar)


def verify_existing_shard(manifest: Path, shard: Path) -> None:
    """Reuse only a shard whose signed manifest and content match this audit."""
    sidecar = shard.with_suffix(".manifest.json")
    verify_signed_file(sidecar)
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    if payload.get("manifest_sha256") != compute_sha256(manifest) or payload.get(
        "shard_sha256"
    ) != compute_sha256(shard):
        raise ValueError(f"PANDA existing shard differs from audited manifest: {shard}")


def publish_inventory(
    official: pd.DataFrame,
    inventory: pd.DataFrame,
    cfg: dict[str, Any],
    raw_hashes: dict[str, dict[str, str | None]],
) -> dict[str, Any]:
    """Publish signed canonical inventory plus aggregate raw and shard provenance."""
    inventory = inventory.copy()
    inventory["image_path"] = inventory.apply(
        lambda row: mounted_path(cfg, row), axis=1
    )
    path = Path(cfg["canonical_inventory_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(path, index=False)
    sign_file(path)
    sidecar = _sidecar_payload(official, inventory, cfg, path, raw_hashes)
    sidecar_path = Path(cfg["sidecar_path"])
    write_json(sidecar_path, sidecar)
    sign_file(sidecar_path)
    return sidecar


def _sidecar_payload(
    official: pd.DataFrame,
    inventory: pd.DataFrame,
    cfg: dict[str, Any],
    path: Path,
    raw_hashes: dict[str, dict[str, str | None]],
) -> dict[str, Any]:
    root = Path(cfg["shard_root"])
    shards = sorted(root.glob("shard=*.manifest.json"))
    if list(root.glob("*.partial")) or len(shards) != int(cfg["shard_count"]):
        raise ValueError("PANDA shards are incomplete or retain partial artifacts")
    for sidecar in shards:
        verify_signed_file(sidecar)
    counts = inventory.patch_label.value_counts()
    return {
        "raw_inventory_sha256": compute_data_hash(raw_hashes),
        "inventory_path": str(path),
        "inventory_sha256": compute_sha256(path),
        "shard_count": int(cfg["shard_count"]),
        "shards": {
            sidecar.name: json.loads(sidecar.read_text(encoding="utf-8"))
            for sidecar in shards
        },
        "cohort_counts": {
            "official_slides": len(official),
            "labelled_slides": inventory.loc[
                inventory.patch_label.isin(("benign", "cancer")), "slide_id"
            ].nunique(),
            "benign_patches": int(counts.get("benign", 0)),
            "cancer_patches": int(counts.get("cancer", 0)),
        },
        "tool_commit": _tool_commit(),
    }


def mounted_path(cfg: dict[str, Any], row: pd.Series) -> str:
    """Return image path as visible from its job-local mounted SqFS shard."""
    index = str(row.patch_id).split("/")[-1]
    return str(
        Path(cfg["shard_mount_root"])
        / f"shard={int(row.shard_index)}"
        / "tiles"
        / str(row.slide_id)
        / f"{index}.jpg"
    )


def _tool_commit() -> str:
    result = subprocess.run(
        ["git", "-C", str(Path(__file__).resolve().parents[5]), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()
