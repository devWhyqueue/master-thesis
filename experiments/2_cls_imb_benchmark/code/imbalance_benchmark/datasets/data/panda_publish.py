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

LOCKED_SLIDES, LOCKED_LABELLED_SLIDES = 10_616, 10_514
LOCKED_BENIGN_PATCHES, LOCKED_CANCER_PATCHES = 3_702_544, 803_785


def materialize_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate required project-owned PANDA materialization paths."""
    cfg = config.get("materialize_panda")
    required = {
        "raw_root",
        "legacy_tiles_dir",
        "scratch_root",
        "shard_root",
        "shard_mount_root",
        "canonical_inventory_path",
        "sidecar_path",
    }
    if not isinstance(cfg, dict):
        raise ValueError(
            f"PANDA materialization fields are missing: {sorted(required)}"
        )
    missing = sorted(required - set(cfg))
    if missing:
        raise ValueError(f"PANDA materialization fields are missing: {missing}")
    return {"shard_count": 48, "jpeg_mae_max": 5.0, **cfg}


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


def publish_shards(inventory: pd.DataFrame, scratch: Path, cfg: dict[str, Any]) -> None:
    """Pack verified shard trees through partial images and atomic publication."""
    for index, shard in inventory.groupby("shard_index", sort=True):
        root = scratch / f"shard={index}"
        move_images(shard, scratch, root)
        manifest = root / "manifest.csv"
        shard.assign(
            image_path=shard.apply(lambda row: mounted_path(cfg, row), axis=1)
        ).to_csv(manifest, index=False)
        destination = Path(cfg["shard_root"]) / f"shard={index}.sqfs"
        if pack_shard(
            root, destination, str(cfg.get("squash_command", "squash-dataset"))
        ):
            write_shard_sidecar(manifest, destination)
        else:
            verify_existing_shard(manifest, destination)
        shutil.rmtree(root)


def move_images(shard: pd.DataFrame, scratch: Path, root: Path) -> None:
    """Move each once-verified JPEG into its project-owned shard tree."""
    for patch_id, slide_id in shard[["patch_id", "slide_id"]].itertuples(
        index=False, name=None
    ):
        index = str(patch_id).split("/")[-1]
        source = scratch / "tiles" / str(slide_id) / f"{index}.jpg"
        target = root / "tiles" / str(slide_id) / f"{index}.jpg"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(source, target)


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
    official: pd.DataFrame, inventory: pd.DataFrame, cfg: dict[str, Any]
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
    sidecar = _sidecar_payload(official, inventory, cfg, path)
    sidecar_path = Path(cfg["sidecar_path"])
    write_json(sidecar_path, sidecar)
    sign_file(sidecar_path)
    return sidecar


def _sidecar_payload(
    official: pd.DataFrame, inventory: pd.DataFrame, cfg: dict[str, Any], path: Path
) -> dict[str, Any]:
    root = Path(cfg["shard_root"])
    shards = sorted(root.glob("shard=*.manifest.json"))
    if list(root.glob("*.partial")) or len(shards) != int(cfg["shard_count"]):
        raise ValueError("PANDA shards are incomplete or retain partial artifacts")
    for sidecar in shards:
        verify_signed_file(sidecar)
    counts = inventory.patch_label.value_counts()
    return {
        "raw_inventory_sha256": compute_data_hash(
            {
                str(row.slide_id): {
                    "image": compute_sha256(Path(str(row.image_path))),
                    "mask": compute_sha256(Path(str(row.mask_path)))
                    if row.has_mask
                    else None,
                }
                for _, row in official.iterrows()
            }
        ),
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
