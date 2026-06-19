import ast
import json
import logging
import os
from collections.abc import Sequence
from typing import cast

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from data.feature_store import (
    feature_for_manifest_row,
    load_feature_cache,
    load_row_feature_cache,
    maybe_feature_store,
    patch_id_for_row,
    patch_row,
    row_level_dataset,
)
from data.filtering import filter_synthetic_rows

logger = logging.getLogger(__name__)


class TCGAUTDatasetImbalanced(Dataset):
    def __init__(
        self,
        dataset_path: str,
        feature_path: str,
        subsets: Sequence[str] | None = None,
        args_path: str | None = None,
        preload_features: bool = False,
        device: str | torch.device = "cpu",
        feature_cache_path: str | None = None,
        split_name: str | None = None,
        include_synthetic: bool = False,
        synthetic_variant_epochs: int | None = None,
    ) -> None:
        super().__init__()
        self.dataset_path = dataset_path
        self.feature_path = feature_path
        self.subsets = subsets
        self.args_path = args_path
        self.preload_features = preload_features
        self.device = device
        self.feature_cache = load_feature_cache(feature_cache_path)
        self.row_feature_cache = load_row_feature_cache(feature_cache_path)
        self.feature_store = maybe_feature_store(feature_path)
        self.split_name = split_name
        self.include_synthetic = include_synthetic
        self.synthetic_variant_epochs = synthetic_variant_epochs
        self.dataset_original = self._load_dataset_structure()
        self.dataset = self._flatten_dataset()
        self.args = self._load_args()
        if self.preload_features:
            logger.info("Loading features into RAM")
            self.dataset = self._preload_features()
        self.features_str_to_int_map = self._class_to_int_map()

    def get_class_sizes(self) -> np.ndarray:
        """Return the number of patch samples per integer class."""
        targets = [self.features_str_to_int_map[t] for t in self.dataset["cancer_type"]]
        return np.bincount(np.array(targets))

    def load_features(
        self,
        slide_ids: Sequence[str],
        targets: Sequence[str],
    ) -> list[dict[str, object]]:
        """Load feature tensors for selected slides."""
        features = []
        for slide_id, target in zip(slide_ids, targets):
            features.extend(self._load_slide_features(slide_id, target))
        return features

    def get_feature_dim(self) -> int:
        """Return the feature vector size."""
        if self.preload_features:
            feature = cast(torch.Tensor, self.dataset.iloc[0]["features"])
            return int(feature.shape[-1])
        slide_id = str(self.dataset.iloc[0]["slide_id"])
        target = str(self.dataset.iloc[0]["cancer_type"])
        feature = cast(
            torch.Tensor, self.load_features([slide_id], [target])[0]["features"]
        )
        return int(feature.shape[-1])

    def get_n_classes(self) -> int:
        """Return the number of classes."""
        return len(self.features_str_to_int_map)

    def get_int_to_class_map(self) -> dict[int, str]:
        """Return the integer-to-class-name mapping."""
        return {value: key for key, value in self.features_str_to_int_map.items()}

    def get_int_targets(self) -> np.ndarray:
        """Return integer targets for every patch sample."""
        targets = self.dataset["cancer_type"].replace(self.features_str_to_int_map)
        return np.asarray(targets, dtype=int)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, object]:
        row = self.dataset.iloc[idx]
        feature = (
            row["features"] if self.preload_features else self._feature_for_row(row)
        )
        target = str(row["cancer_type"])
        return {
            "slide_id": row["slide_id"],
            "features": cast(torch.Tensor, feature).to(self.device),
            "target_str": target,
            "target": self.features_str_to_int_map[target],
            "patch_id": row["patch_id"],
        }

    def _load_dataset_structure(self) -> pd.DataFrame:
        dataset_original = pd.read_csv(self.dataset_path)
        if self.split_name is not None and "split" in dataset_original.columns:
            mask = dataset_original["split"] == self.split_name
            dataset_original = dataset_original.loc[mask].copy()
        dataset_original = filter_synthetic_rows(
            dataset_original, self.include_synthetic, self.synthetic_variant_epochs
        )
        if "patch_ids" in dataset_original.columns and isinstance(
            dataset_original["patch_ids"].iloc[0], str
        ):
            dataset_original["patch_ids"] = dataset_original["patch_ids"].apply(
                ast.literal_eval
            )
        if self.subsets is not None:
            subset = dataset_original[dataset_original["slide_id"].isin(self.subsets)]
            return cast(pd.DataFrame, subset)
        return dataset_original

    def _flatten_dataset(self) -> pd.DataFrame:
        if "patch_ids" not in self.dataset_original.columns:
            return row_level_dataset(self.dataset_original)
        rows = []
        for _, row in self.dataset_original.iterrows():
            for patch_id in cast(Sequence[str], row["patch_ids"]):
                rows.append(
                    patch_row(
                        str(row["cancer_type"]),
                        str(row["slide_id"]),
                        patch_id,
                    )
                )
        return pd.DataFrame(rows)

    def _load_args(self) -> dict[str, object]:
        if self.args_path is None:
            return {}
        with open(self.args_path) as file:
            return json.load(file)

    def _preload_features(self) -> pd.DataFrame:
        if "feature_path" in self.dataset.columns:
            dataset = self.dataset.copy()
            dataset["features"] = dataset.apply(
                lambda row: feature_for_manifest_row(
                    row, self.row_feature_cache, self.feature_cache
                ),
                axis=1,
            )
            return dataset
        features = []
        class_names = list(dict.fromkeys(self.dataset["cancer_type"].to_list()))
        for class_name in class_names:
            slides = self.dataset[self.dataset["cancer_type"] == class_name]["slide_id"]
            slide_ids = [str(slide_id) for slide_id in dict.fromkeys(list(slides))]
            features.extend(
                self.load_features(slide_ids, [str(class_name) for _ in slide_ids])
            )
        return cast(
            pd.DataFrame,
            self.dataset.merge(
                pd.DataFrame(features), on=["slide_id", "patch_id"], how="inner"
            ),
        )

    def _class_to_int_map(self) -> dict[str, int]:
        return {
            class_name: index
            for index, class_name in enumerate(
                sorted(self.dataset["cancer_type"].unique())
            )
        }

    def _load_slide_features(
        self, slide_id: str, target: str
    ) -> list[dict[str, object]]:
        if "feature_path" in self.dataset_original.columns:
            rows = self.dataset_original[
                (self.dataset_original["slide_id"] == slide_id)
                & (self.dataset_original["cancer_type"] == target)
            ]
            return [
                {
                    "features": feature_for_manifest_row(
                        row, self.row_feature_cache, self.feature_cache
                    ),
                    "patch_id": patch_id_for_row(row),
                    "slide_id": slide_id,
                }
                for _, row in rows.iterrows()
            ]
        if self.feature_store is not None:
            patch_ids = self._patch_ids_for_slide(slide_id)
            return [
                {
                    "features": self.feature_store.load_patch_feature(
                        slide_id, patch_ids, patch_id
                    ),
                    "patch_id": patch_id,
                    "slide_id": slide_id,
                }
                for patch_id in patch_ids
            ]
        tensor = torch.load(os.path.join(self.feature_path, target, f"{slide_id}.pt"))
        patch_ids = self._patch_ids_for_slide(slide_id)
        with open(os.path.join(self.feature_path, target, f"{slide_id}.json")) as file:
            patch_mapping = json.load(file)
        indices = [patch_mapping.index(f"{patch_id}.jpg") for patch_id in patch_ids]
        return [
            {"features": feature, "patch_id": patch_ids[index], "slide_id": slide_id}
            for index, feature in enumerate(tensor[indices])
        ]

    def _patch_ids_for_slide(self, slide_id: str) -> list[str]:
        patch_ids = self.dataset_original[
            self.dataset_original["slide_id"] == slide_id
        ]["patch_ids"].tolist()[0]
        return [str(patch_id) for patch_id in patch_ids]

    def _feature_for_row(self, row: pd.Series) -> torch.Tensor:
        if "features" in row and row["features"] is not None:
            return cast(torch.Tensor, row["features"])
        if "feature_path" in row or "feature_index" in row:
            return feature_for_manifest_row(
                row, self.row_feature_cache, self.feature_cache
            )
        features = self.load_features([str(row["slide_id"])], [str(row["cancer_type"])])
        feature = [
            item["features"] for item in features if item["patch_id"] == row["patch_id"]
        ][0]
        return cast(torch.Tensor, feature)
