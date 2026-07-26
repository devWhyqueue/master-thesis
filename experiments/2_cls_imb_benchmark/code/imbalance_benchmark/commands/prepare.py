from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from imbalance_benchmark.common import (
    ensure_dirs,
    load_config,
    split_indices,
    split_paths,
)
from imbalance_benchmark.construction import split_cases
from imbalance_benchmark.datasets import build_manifest
from imbalance_benchmark.datasets.bracs import discover_slides, tile_slide
from imbalance_benchmark.datasets.bracs.audit import validate_tile_manifest
from imbalance_benchmark.datasets.features import (
    attach_extracted_features,
    resolve_feature_provenance,
)
from imbalance_benchmark.datasets.features.provenance_lock import (
    write_prepared_feature_provenance,
)
from imbalance_benchmark.manifest.seeds import derive_seed

__all__ = ["cmd_prepare", "cmd_tile_wsi", "cmd_tile_wsi_reduce"]


SYNTHETIC_PATCHES_PER_SLIDE = 30


def _synthetic_manifest(paths: dict[str, Path]) -> pd.DataFrame:
    """Build a synthetic manifest for the smoke workflow (no external datasets).

    20 patients/class survives the 70/15/15 split with >=10 in train, the
    minimum the pilot's 10% patient-contribution cap can ever satisfy
    (manifest/pilot_training.py:patch_pilot_caps_hold). One row per patch
    (feature_index into the slide's multi-row tensor), mirroring the real
    dataset builders' expansion (e.g. datasets/tcga_ut.py:_expand_chunks) --
    ImbalanceDataset requires an explicit feature_index for multi-row tensors.
    """
    rows = []
    for cls in ["class_A", "class_B", "class_C", "class_D"]:
        for p_idx in range(20):
            p_id = f"PAT_{cls}_{p_idx}"
            for s_idx in range(2):
                s_id = f"SLIDE_{p_id}_{s_idx}"
                f_path = paths["data"] / f"{s_id}.pt"
                if not f_path.exists():
                    torch.save(torch.randn(SYNTHETIC_PATCHES_PER_SLIDE, 2560), f_path)
                rows.extend(
                    {
                        "case_id": p_id,
                        "slide_id": s_id,
                        "cancer_type": cls,
                        "feature_path": str(f_path),
                        "feature_index": patch_idx,
                    }
                    for patch_idx in range(SYNTHETIC_PATCHES_PER_SLIDE)
                )
    return pd.DataFrame(rows)


def _base_manifest(config: dict[str, object], paths: dict[str, Path]) -> pd.DataFrame:
    """Build the eligible feature manifest once before deriving patient splits."""
    dataset_cfg = config.get("dataset", {})
    if not isinstance(dataset_cfg, dict):
        raise ValueError("dataset config must be a mapping")
    dataset_name = dataset_cfg.get("name", "synthetic")
    if dataset_name == "synthetic":
        return _synthetic_manifest(paths)
    feature_cfg = config.get("feature_extraction", {})
    if not isinstance(feature_cfg, dict):
        raise ValueError("feature_extraction config must be a mapping")
    resolve_feature_provenance(feature_cfg)
    df = build_manifest(config)
    if "image_path" not in df.columns or "feature_path" in df.columns:
        return df
    return attach_extracted_features(
        df, paths["data"] / "features" / str(dataset_name), feature_cfg
    )


def cmd_prepare(args: argparse.Namespace) -> None:
    """Create exactly three disjoint patient-split manifests from one eligible pool."""
    config = load_config(args.config)
    base_paths = ensure_dirs(config)
    df = _base_manifest(config, base_paths)
    for index in split_indices(args.split_index):
        paths = split_paths(base_paths, index)
        split_df = split_cases(
            df, seed=derive_seed(args.seed, f"patient_split_{index}")
        )
        split_df.to_csv(paths["data"] / "manifest.csv", index=False)
        split_df.drop_duplicates("slide_id").to_csv(
            paths["data"] / "slide_manifest.csv", index=False
        )
        write_prepared_feature_provenance(config, paths["data"])


def _wsi_tile_root(config: dict[str, object]) -> Path:
    dataset_cfg = config["dataset"]
    if not isinstance(dataset_cfg, dict):
        raise ValueError("dataset config must be a mapping")
    return Path(dataset_cfg["wsi_tile_root"])


def cmd_tile_wsi(args: argparse.Namespace) -> None:
    """Tile one shard of BRACS slides and write per-slide partial manifests."""
    config = load_config(args.config)
    dataset_cfg = config["dataset"]
    if not isinstance(dataset_cfg, dict):
        raise ValueError("dataset config must be a mapping")
    slides = discover_slides(Path(dataset_cfg["root"]))
    slide_ids = sorted(slides)
    start = args.slide_index * args.shard_size
    selected = slide_ids[start : start + args.shard_size]
    if not selected:
        raise ValueError(
            f"slide-index {args.slide_index} covers no slides "
            f"({len(slide_ids)} discovered, shard-size {args.shard_size})"
        )
    tile_root = _wsi_tile_root(config)
    partial_dir = tile_root / "_partials"
    partial_dir.mkdir(parents=True, exist_ok=True)
    for slide_id in selected:
        frame = tile_slide(slides[slide_id], slide_id, tile_root)
        frame.to_csv(partial_dir / f"{slide_id}.csv", index=False)


def cmd_tile_wsi_reduce(args: argparse.Namespace) -> None:
    """Concatenate per-slide partials into the validated ``tile_manifest.csv``."""
    config = load_config(args.config)
    dataset_cfg = config["dataset"]
    if not isinstance(dataset_cfg, dict):
        raise ValueError("dataset config must be a mapping")
    tile_root = _wsi_tile_root(config)
    manifest_path = Path(
        dataset_cfg.get("wsi_tile_manifest", tile_root / "tile_manifest.csv")
    )
    expected_slides = int(dataset_cfg.get("expected_wsi_count", 547))
    partials = sorted((tile_root / "_partials").glob("*.csv"))
    if not partials:
        raise FileNotFoundError(f"No BRACS WSI tile partials found under {tile_root}")
    frame = pd.concat((pd.read_csv(path) for path in partials), ignore_index=True)
    validate_tile_manifest(frame, expected_slides)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(manifest_path, index=False)
