"""Specialized patch-feature trainers outside the standard CE loop."""

from __future__ import annotations

import torch

from scripts.patch_feature.cfal import train_cfal_model
from scripts.patch_feature.divide_conquer_train import train_divide_conquer_model
from scripts.patch_feature.training import PatchFeatureDataset


def fit_special_patch_method(
    method: str,
    train_set: PatchFeatureDataset,
    class_names: list[str],
    settings: dict,
    device: torch.device,
    seed: int,
    tuning_params: dict[str, float],
) -> tuple[torch.nn.Module, dict[str, object] | None] | None:
    """Train CFAL or divide-and-conquer when method matches; else return None."""
    if method == "patch_feature_cfal":
        return (
            train_cfal_model(
                train_set, len(class_names), settings, device, seed, tuning_params
            ),
            None,
        )
    if method == "patch_feature_divide_conquer":
        model, diagnostics = train_divide_conquer_model(
            train_set,
            class_names,
            len(class_names),
            settings,
            device,
            seed,
            tuning_params,
        )
        return model, diagnostics
    return None
