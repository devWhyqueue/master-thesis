from __future__ import annotations

from pathlib import Path

import torch

from imbalance_benchmark.datasets.data.bag import BagFeatureDataset, bag_collate
from imbalance_benchmark.datasets.data.common import slide_level_identity
from imbalance_benchmark.datasets.data.patch import ImbalanceDataset
from imbalance_benchmark.datasets.features import load_feature_row

__all__ = [
    "load_feature_row",
    "ImbalanceDataset",
    "BagFeatureDataset",
    "bag_collate",
    "slide_level_identity",
    "load_training_dataset",
    "TrainDataset",
]


def load_training_dataset(
    manifest_path: str | Path,
    is_mil: bool,
    split_name: str | None = None,
    device: str | torch.device = "cpu",
    class_names: list[str] | None = None,
) -> TrainDataset:
    """Load one regime-appropriate CPU-resident frozen-feature dataset."""
    if not is_mil:
        return ImbalanceDataset(manifest_path, split_name, device, class_names)
    return BagFeatureDataset(
        manifest_path, split_name, device=device, class_names=class_names
    )


TrainDataset = ImbalanceDataset | BagFeatureDataset
