"""Resolve cached patch features and manifests."""

from __future__ import annotations

import os
from pathlib import Path
from typing import cast

import pandas as pd
import torch

from data.feature_store import load_feature_row, load_slide_features


def patch_row(cancer_type: str, slide_id: str, patch_id: str) -> dict[str, str]:
    """Build one flattened patch manifest row."""
    return {"cancer_type": cancer_type, "slide_id": slide_id, "patch_id": patch_id}


def row_level_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize a row-level manifest to include patch_id."""
    dataset = frame.copy()
    if "patch_id" not in dataset.columns:
        dataset["patch_id"] = dataset.apply(patch_id_for_row, axis=1)
    return cast(pd.DataFrame, dataset)


def patch_id_for_row(row: pd.Series) -> str:
    """Infer a patch identifier from common manifest columns."""
    for column in [
        "patch_id",
        "feature_id",
        "feature_index",
        "image_path",
        "feature_path",
    ]:
        value = row.get(column)
        if value is not None and pd.notna(value):
            return os.path.splitext(os.path.basename(str(row[column])))[0]
    return str(row["slide_id"])


def feature_for_manifest_row(
    row: pd.Series,
    row_feature_cache: dict[tuple[str, int], torch.Tensor] | None,
    feature_cache: dict[str, torch.Tensor] | None,
) -> torch.Tensor:
    """Resolve one manifest row to a feature vector."""
    path = str(row["feature_path"])
    index_value = row.get("feature_index")
    if index_value is not None and pd.notna(index_value):
        row_index = int(index_value)
        if row_feature_cache is not None:
            cached = row_feature_cache.get((path, row_index))
            if cached is not None:
                return cached
        return load_feature_row(path, row_index)
    if feature_cache is not None and path in feature_cache:
        return feature_cache[path]
    features = load_slide_features(path)
    if features.shape[0] == 1:
        return features[0].squeeze()
    return features[0].squeeze()


def load_feature_cache(path: str | None) -> dict[str, torch.Tensor] | None:
    """Load a precomputed feature cache keyed by feature path."""
    if path is None:
        return None
    payload = torch.load(path, map_location="cpu")
    feature_paths = [str(feature_path) for feature_path in payload["feature_paths"]]
    features = cast(torch.Tensor, payload["features"])
    return dict(zip(feature_paths, features, strict=True))


def load_row_feature_cache(
    path: str | None,
) -> dict[tuple[str, int], torch.Tensor] | None:
    """Load a row-level cache keyed by (feature_path, feature_index)."""
    if path is None or not Path(path).is_file():
        return None
    payload = torch.load(path, map_location="cpu")
    paths = payload.get("feature_paths")
    indices = payload.get("feature_indices")
    features = payload.get("features")
    if paths is None or indices is None or features is None:
        return None
    stacked = cast(torch.Tensor, features)
    pairs = zip(paths, indices, strict=True)
    return {
        (str(feature_path), int(feature_index)): stacked[row_index]
        for row_index, (feature_path, feature_index) in enumerate(pairs)
    }
