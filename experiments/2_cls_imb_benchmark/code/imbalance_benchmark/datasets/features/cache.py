from __future__ import annotations

from functools import lru_cache

import torch

__all__ = ["load_feature_row", "load_slide_features"]


@lru_cache(maxsize=512)
def load_slide_features(path: str) -> torch.Tensor:
    """Load a feature tensor and normalize to (n_instances, dim)."""
    tensor = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(tensor, dict):
        class_token, mean_token = tensor.get("cls"), tensor.get("mean_patch")
        features = (
            torch.cat([class_token, mean_token], dim=-1).float()
            if class_token is not None and mean_token is not None
            else next(
                value for value in tensor.values() if torch.is_tensor(value)
            ).float()
        )
    else:
        features = tensor.float()
    if features.ndim == 1:
        return features.unsqueeze(0)
    if features.ndim > 2:
        return features.reshape(-1, features.shape[-1])
    return features


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
