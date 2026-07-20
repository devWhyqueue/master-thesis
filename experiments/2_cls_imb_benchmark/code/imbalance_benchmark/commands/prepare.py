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
from imbalance_benchmark.datasets.features import (
    attach_extracted_features,
    resolve_feature_provenance,
)
from imbalance_benchmark.datasets.feature_provenance import (
    VIRCHOW2_REVISION,
    VIRCHOW2_WEIGHTS_SHA256,
)
from imbalance_benchmark.manifest.seeds import derive_seed

__all__ = ["cmd_prepare"]


def _synthetic_manifest(paths: dict[str, Path]) -> pd.DataFrame:
    """Build a synthetic manifest for the smoke workflow (no external datasets)."""
    rows = []
    for cls in ["class_A", "class_B", "class_C", "class_D"]:
        for p_idx in range(12):
            p_id = f"PAT_{cls}_{p_idx}"
            for s_idx in range(2):
                s_id = f"SLIDE_{p_id}_{s_idx}"
                f_path = paths["data"] / f"{s_id}.pt"
                if not f_path.exists():
                    torch.save(torch.randn(30, 2560), f_path)
                rows.append(
                    {
                        "case_id": p_id,
                        "slide_id": s_id,
                        "cancer_type": cls,
                        "feature_path": str(f_path),
                    }
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
        df,
        paths["data"] / "features" / str(dataset_name),
        model_name=str(feature_cfg.get("model_name", "hf-hub:paige-ai/Virchow2")),
        batch_size=int(feature_cfg.get("batch_size", 64)),
        dtype=str(feature_cfg.get("dtype", "float16")),
        revision=str(feature_cfg.get("revision", VIRCHOW2_REVISION)),
        weights_sha256=str(feature_cfg.get("weights_sha256", VIRCHOW2_WEIGHTS_SHA256)),
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
