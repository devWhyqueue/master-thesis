from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
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
from imbalance_benchmark.manifest.seeds import derive_seed

__all__ = ["cmd_prepare"]


def _apply_patch_evidence_cap(
    frame: pd.DataFrame, patch_per_slide_cap: int | None, seed: int
) -> pd.DataFrame:
    """Deterministically cap eligible patch evidence before patient splitting."""
    if patch_per_slide_cap is None:
        return frame
    if patch_per_slide_cap < 1:
        raise ValueError("evidence.patch_per_slide_cap must be positive")
    selected = []
    for slide_id, group in frame.groupby("slide_id", sort=False):
        if len(group) <= patch_per_slide_cap:
            selected.append(group)
            continue
        digest = hashlib.sha256(f"{seed}:{slide_id}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
        positions = np.sort(rng.choice(len(group), patch_per_slide_cap, replace=False))
        selected.append(group.iloc[positions])
    return pd.concat(selected, ignore_index=True)


def _apply_evidence_controls(
    frame: pd.DataFrame, config: dict[str, object], seed: int
) -> pd.DataFrame:
    """Apply regime-specific evidence limits to the eligible pool once."""
    dataset = config.get("dataset", {})
    if not isinstance(dataset, dict) or dataset.get("regime", "patch") != "patch":
        return frame
    evidence = config.get("evidence", {})
    if not isinstance(evidence, dict):
        raise ValueError("evidence config must be a mapping")
    cap = evidence.get("patch_per_slide_cap")
    return _apply_patch_evidence_cap(frame, None if cap is None else int(cap), seed)


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
    )


def cmd_prepare(args: argparse.Namespace) -> None:
    """Create exactly three disjoint patient-split manifests from one eligible pool."""
    config = load_config(args.config)
    base_paths = ensure_dirs(config)
    df = _apply_evidence_controls(
        _base_manifest(config, base_paths),
        config,
        derive_seed(args.seed, "instance_selection"),
    )
    for index in split_indices(args.split_index):
        paths = split_paths(base_paths, index)
        split_df = split_cases(
            df, seed=derive_seed(args.seed, f"patient_split_{index}")
        )
        split_df.to_csv(paths["data"] / "manifest.csv", index=False)
        split_df.drop_duplicates("slide_id").to_csv(
            paths["data"] / "slide_manifest.csv", index=False
        )
