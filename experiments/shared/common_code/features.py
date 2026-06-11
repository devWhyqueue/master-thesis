"""Feature tensor loading helpers."""

from __future__ import annotations

import torch


def feature_to_bag(path: str, max_instances: int | None) -> torch.Tensor:
    """Load a feature file and normalize it to a 2D bag tensor."""
    tensor = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(tensor, dict):
        tensor = next(value for value in tensor.values() if torch.is_tensor(value))
    features = tensor.float()
    if features.ndim == 1:
        features = features.unsqueeze(0)
    if features.ndim > 2:
        features = features.reshape(-1, features.shape[-1])
    if max_instances and len(features) > max_instances:
        indices = torch.linspace(0, len(features) - 1, max_instances).long()
        features = features.index_select(0, indices)
    return features


load_slide_features = feature_to_bag
