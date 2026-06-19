"""Specialized patch-feature trainers outside the standard CE loop."""

from __future__ import annotations

from importlib import import_module
from typing import Callable, cast

import torch

from scripts.modeling.patch_feature.training import PatchFeatureDataset

SpecializedTrainer = Callable[..., torch.nn.Module]

train_cfal_model = cast(
    SpecializedTrainer,
    getattr(import_module("scripts.modeling.patch_feature.cfal"), "train_cfal_model"),
)
train_oko_model = cast(
    SpecializedTrainer,
    getattr(import_module("scripts.modeling.patch_feature.oko"), "train_oko_model"),
)


def fit_special_patch_method(
    method: str,
    train_set: PatchFeatureDataset,
    class_names: list[str],
    settings: dict,
    device: torch.device,
    seed: int,
    tuning_params: dict[str, float],
) -> tuple[torch.nn.Module, dict[str, object] | None] | None:
    """Train patch-feature methods with specialized optimization loops."""
    if method == "patch_feature_cfal":
        return (
            train_cfal_model(
                train_set, len(class_names), settings, device, seed, tuning_params
            ),
            None,
        )
    if method == "patch_feature_oko":
        return train_oko_model(
            train_set, len(class_names), settings, device, seed, tuning_params
        ), None
    return None
