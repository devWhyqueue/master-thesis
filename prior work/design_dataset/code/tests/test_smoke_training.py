"""Local smoke tests for patch features and WSI bag cache on synthetic data."""

import json
from pathlib import Path

import pandas as pd
import torch

from tcga_ut_imbalanced.data.full_scale_sampling import write_constructed_outputs
from tcga_ut_imbalanced.training.constructed_wsi_cache import _write_split_cache
from tcga_ut_imbalanced.training.constructed_wsi_data import ConstructedBagDataset


def _feature_dir(tmp_path: Path, slide_id: str, patch_ids: list[str]) -> Path:
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    vectors = torch.stack([torch.randn(2560) for _ in patch_ids])
    torch.save(vectors, feature_dir / f"{slide_id}_0.pt")
    return feature_dir


def test_patch_manifest_feature_dim(tmp_path) -> None:
    slide_id = "TCGA-XX-0001"
    patch_ids = ["0_0", "0_9", "1_0"]
    feature_dir = _feature_dir(tmp_path, slide_id, patch_ids)
    frame = pd.DataFrame(
        [{"cancer_type": "a", "slide_id": slide_id, "patch_ids": patch_ids}]
    )
    out = tmp_path / "constructed"
    write_constructed_outputs(
        {"train": frame, "validation": frame, "test": frame},
        {"a": 1},
        ["a"],
        str(out),
        {"seed": 0},
        feature_dir=str(feature_dir),
    )
    manifest = pd.read_csv(out / "manifest_splits.csv")
    row = manifest.iloc[0]
    tensor = torch.load(row["feature_path"], map_location="cpu")
    vector = tensor[int(row["feature_index"])] if tensor.ndim == 2 else tensor
    assert int(vector.shape[-1]) == 2560


def test_wsi_bag_cache_and_dataset(tmp_path) -> None:
    slide_id = "TCGA-XX-0001"
    patch_ids = ["0_0", "0_9", "1_0"]
    feature_dir = _feature_dir(tmp_path, slide_id, patch_ids)
    frame = pd.DataFrame(
        [{"cancer_type": "a", "slide_id": slide_id, "patch_ids": patch_ids}]
    )
    out = tmp_path / "constructed"
    write_constructed_outputs(
        {"train": frame, "validation": frame, "test": frame},
        {"a": 1},
        ["a"],
        str(out),
        {"seed": 0},
        feature_dir=str(feature_dir),
    )
    manifest = pd.read_csv(out / "manifest_splits.csv")
    cache_dir = out / "wsi_bag_cache"
    cache_dir.mkdir()
    train_frame = manifest[manifest["split"] == "train"].copy()
    _write_split_cache(train_frame, cache_dir, "train")
    meta = json.loads((cache_dir / "train_meta.json").read_text(encoding="utf-8"))
    assert meta["feature_dim"] == 2560
    dataset = ConstructedBagDataset(
        train_frame,
        {"a": 0},
        max_instances=30,
        cache_dir=str(cache_dir),
        split="train",
    )
    bag, label = dataset[0]
    assert bag.shape[-1] == 2560
    assert label == 0
