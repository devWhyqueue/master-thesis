from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from imbalance_benchmark.common import ensure_dirs, load_config
from imbalance_benchmark.construction import split_cases
from imbalance_benchmark.datasets import build_manifest
from imbalance_benchmark.datasets.features import attach_extracted_features

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


def cmd_prepare(args: argparse.Namespace) -> None:
    """Validate the configured dataset, create patient-disjoint splits, and extract Virchow2 features."""
    config = load_config(args.config)
    paths = ensure_dirs(config)
    dataset_cfg = config.get("dataset", {})
    dataset_name = dataset_cfg.get("name", "synthetic")
    if dataset_name == "synthetic":
        df = split_cases(_synthetic_manifest(paths), seed=args.seed)
    else:
        df = build_manifest(config)
        if "image_path" in df.columns and "feature_path" not in df.columns:
            feature_cfg = config.get("feature_extraction", {})
            df = attach_extracted_features(
                df,
                paths["data"] / "features" / dataset_name,
                model_name=feature_cfg.get("model_name", "hf-hub:paige-ai/Virchow2"),
                batch_size=int(feature_cfg.get("batch_size", 64)),
                dtype=feature_cfg.get("dtype", "float16"),
            )
    df.to_csv(paths["data"] / "manifest.csv", index=False)
    df.drop_duplicates("slide_id").to_csv(
        paths["data"] / "slide_manifest.csv", index=False
    )
