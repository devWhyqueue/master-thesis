from __future__ import annotations

from functools import lru_cache

import torch

from imbalance_benchmark.datasets.feature_provenance import (
    load_stored_feature_tensor,
)

__all__ = ["load_feature_row", "load_slide_features"]


@lru_cache(maxsize=512)
def load_slide_features(path: str) -> torch.Tensor:
    """Load a feature tensor and normalize to float (n_instances, dim)."""
    return load_stored_feature_tensor(path).float()


def load_feature_row(path: str, index: int | None = None) -> torch.Tensor:
    """Load one feature vector; a multi-row tensor requires an explicit index."""
    features = load_slide_features(path)
    if index is not None:
        return features[int(index)].squeeze()
    if features.shape[0] == 1:
        return features[0].squeeze()
    raise ValueError(
        f"Feature file {path} has {features.shape[0]} rows; "
        "provide feature_index for multi-row tensors."
    )
