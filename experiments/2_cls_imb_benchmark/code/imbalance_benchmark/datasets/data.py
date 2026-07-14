from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, cast
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from imbalance_benchmark.datasets.features import load_slide_features

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
    bag_kwargs: dict[str, int] | None = None,
) -> TrainDataset:
    """Load one regime-appropriate dataset with its frozen evidence controls."""
    if not is_mil:
        return ImbalanceDataset(manifest_path, split_name, device, class_names)
    controls = bag_kwargs or {}
    return BagFeatureDataset(
        manifest_path,
        split_name,
        max_instances=controls.get("max_instances", 500),
        instance_selection_seed=controls.get("instance_selection_seed", 0),
        device=device,
        class_names=class_names,
    )


def _class_names(values: pd.Series) -> list[str]:
    """Return the canonical semantic class order used by every data split."""
    names = sorted(values.astype(str).unique().tolist())
    return (
        sorted(names, key=lambda name: int(name.removeprefix("ISUP")))
        if names and all(name.startswith("ISUP") for name in names)
        else names
    )


def _validate_class_names(df: pd.DataFrame, class_names: list[str]) -> None:
    unexpected = sorted(set(df["cancer_type"].astype(str)) - set(class_names))
    if unexpected:
        raise ValueError(
            f"Manifest contains classes absent from the locked target: {unexpected}"
        )


def slide_level_identity(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse feature chunks to one labelled row per slide after consistency checks."""
    label_counts = cast(
        pd.Series, df.groupby("slide_id", sort=False)["cancer_type"].nunique()
    )
    mixed = cast(pd.Series, label_counts[label_counts != 1])
    if not mixed.empty:
        mixed_slides = list(cast(Any, mixed.index))[:5]
        raise ValueError(
            f"Each WSI must have exactly one class; mixed labels: {mixed_slides}"
        )
    case_counts = cast(
        pd.Series, df.groupby("slide_id", sort=False)["case_id"].nunique()
    )
    inconsistent_cases = cast(pd.Series, case_counts[case_counts != 1])
    if not inconsistent_cases.empty:
        inconsistent_slides = list(cast(Any, inconsistent_cases.index))[:5]
        raise ValueError(
            f"Each WSI must have exactly one patient; inconsistent slides: {inconsistent_slides}"
        )
    return df.drop_duplicates("slide_id", keep="first").reset_index(drop=True)


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
        class_names: list[str] | None = None,
    ) -> None:
        """Initialize the ImbalanceDataset."""
        super().__init__()
        self.device = device
        df = cast(pd.DataFrame, pd.read_csv(manifest_path))
        if split_name is not None and "split" in df.columns:
            df = cast(
                pd.DataFrame, df[df["split"] == split_name].reset_index(drop=True)
            )
        self.df = df
        self.classes = (
            list(class_names)
            if class_names is not None
            else _class_names(cast(pd.Series, self.df["cancer_type"]))
        )
        _validate_class_names(self.df, self.classes)
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}

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
        instance_selection_seed: int = 0,
        device: str | torch.device = "cpu",
        class_names: list[str] | None = None,
    ) -> None:
        """Initialize the BagFeatureDataset."""
        super().__init__()
        self.device = device
        self.max_instances = max_instances
        self.instance_selection_seed = instance_selection_seed
        df = cast(pd.DataFrame, pd.read_csv(manifest_path))
        if split_name is not None and "split" in df.columns:
            df = cast(
                pd.DataFrame, df[df["split"] == split_name].reset_index(drop=True)
            )
        # A TCGA-UT slide may be represented by several feature chunks.  Keep
        # all rows here so __getitem__ can concatenate them before capping.
        slide_rows = slide_level_identity(df)
        self.df = cast(
            pd.DataFrame,
            (
                df.groupby("slide_id", sort=False)
                .agg({"case_id": "first", "cancer_type": "first", "feature_path": list})
                .reset_index()
            ),
        )
        self.classes = (
            list(class_names)
            if class_names is not None
            else _class_names(cast(pd.Series, slide_rows["cancer_type"]))
        )
        _validate_class_names(self.df, self.classes)
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}

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
        paths = list(dict.fromkeys(str(path) for path in row["feature_path"]))
        features = torch.cat([load_slide_features(path) for path in paths], dim=0)
        if self.max_instances is not None and len(features) > self.max_instances:
            if self.instance_selection_seed == 0:
                selected = torch.linspace(
                    0, len(features) - 1, self.max_instances
                ).long()
            else:
                digest = hashlib.sha256(
                    f"{self.instance_selection_seed}:{row['slide_id']}".encode()
                ).digest()
                rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
                selected = torch.as_tensor(
                    np.sort(
                        rng.choice(len(features), self.max_instances, replace=False)
                    ),
                    dtype=torch.long,
                )
            features = features.index_select(0, selected)
        return features.to(self.device), self.class_to_idx[str(row["cancer_type"])]


def bag_collate(
    items: list[tuple[torch.Tensor, int]],
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Collate variable-size bags without padding."""
    bags, labels = zip(*items, strict=False)
    return list(bags), torch.tensor(labels, dtype=torch.long)


TrainDataset = ImbalanceDataset | BagFeatureDataset
