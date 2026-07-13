from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from imbalance_benchmark.datasets.features import load_slide_features

logger = logging.getLogger(__name__)

__all__ = ["load_feature_row", "ImbalanceDataset", "BagFeatureDataset", "bag_collate"]


def load_feature_row(path: str, index: int | None = None) -> torch.Tensor:
    """Load a feature tensor and return the vector at index or squeeze single vector."""
    features = load_slide_features(path)
    if index is not None:
        return features[int(index)].squeeze()
    return features[0].squeeze()


class ImbalanceDataset(Dataset):
    """Unified dataset class for loading frozen Virchow2 patch features."""

    def __init__(
        self,
        manifest_path: str | Path,
        split_name: str | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        """Initialize the ImbalanceDataset."""
        super().__init__()
        self.device = device
        df = pd.read_csv(manifest_path)
        if split_name is not None and "split" in df.columns:
            df = df[df["split"] == split_name].reset_index(drop=True)
        self.df = df
        self.classes = sorted(list(set(self.df["cancer_type"])))
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}

    def get_class_sizes(self) -> np.ndarray:
        """Get class sample sizes."""
        targets = [self.class_to_idx[name] for name in self.df["cancer_type"]]
        return np.bincount(targets, minlength=len(self.classes))

    def get_n_classes(self) -> int:
        """Get number of classes."""
        return len(self.classes)

    def get_int_targets(self) -> np.ndarray:
        """Get integer array of targets."""
        return np.array(
            [self.class_to_idx[name] for name in self.df["cancer_type"]], dtype=int
        )

    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a single patch sample."""
        row = self.df.iloc[idx]
        feature_path = str(row["feature_path"])
        feature_index = (
            int(row["feature_index"])
            if "feature_index" in row and pd.notna(row["feature_index"])
            else None
        )
        feature = load_feature_row(feature_path, feature_index)
        target_str = str(row["cancer_type"])
        return {
            "slide_id": row["slide_id"],
            "features": feature.to(self.device),
            "target_str": target_str,
            "target": self.class_to_idx[target_str],
            "patch_id": row.get("patch_id", f"{row['slide_id']}_{idx}"),
        }


class BagFeatureDataset(Dataset):
    """Multi-class WSI bag dataset for Weakly-Supervised MIL."""

    def __init__(
        self,
        manifest_path: str | Path,
        split_name: str | None = None,
        max_instances: int = 500,
        device: str | torch.device = "cpu",
    ) -> None:
        """Initialize the BagFeatureDataset."""
        super().__init__()
        self.device = device
        self.max_instances = max_instances
        df = pd.read_csv(manifest_path)
        if split_name is not None and "split" in df.columns:
            df = df[df["split"] == split_name].reset_index(drop=True)
        self.df = df.groupby("slide_id").first().reset_index()
        self.classes = sorted(list(set(self.df["cancer_type"])))
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}

    def get_class_sizes(self) -> np.ndarray:
        """Get class bag sizes."""
        targets = [self.class_to_idx[name] for name in self.df["cancer_type"]]
        return np.bincount(targets, minlength=len(self.classes))

    def get_n_classes(self) -> int:
        """Get number of classes."""
        return len(self.classes)

    def get_int_targets(self) -> np.ndarray:
        """Get integer array of bag targets."""
        return np.array(
            [self.class_to_idx[name] for name in self.df["cancer_type"]], dtype=int
        )

    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Get a single bag sample."""
        row = self.df.iloc[idx]
        features = load_slide_features(str(row["feature_path"]))
        if self.max_instances is not None and len(features) > self.max_instances:
            features = features.index_select(
                0, torch.linspace(0, len(features) - 1, self.max_instances).long()
            )
        return features.to(self.device), self.class_to_idx[str(row["cancer_type"])]


def bag_collate(
    items: list[tuple[torch.Tensor, int]],
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Collate variable-size bags without padding."""
    bags, labels = zip(*items, strict=False)
    return list(bags), torch.tensor(labels, dtype=torch.long)
