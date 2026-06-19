"""Adapter datasets for patch-feature specialized trainers in common_code."""

from __future__ import annotations

import argparse
from typing import cast

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from common_code.losses.oko import train_oko_model
from common_code.wsi.cfal import train_cfal_model
from common_code.wsi.divide_conquer.train import train_divide_conquer_model
from data.dataset import TCGAUTDatasetImbalanced


class PreloadedPatchFeatureDataset(Dataset):
    """Torch dataset view of a preloaded patch-feature training set."""

    def __init__(self, source: TCGAUTDatasetImbalanced) -> None:
        if not source.preload_features:
            raise ValueError("Specialized trainers require --preload-features.")
        frame = source.dataset
        self.features = np.stack(
            [
                cast(torch.Tensor, feature).float().cpu().numpy()
                for feature in frame["features"].tolist()
            ]
        )
        self.indices = np.arange(len(self.features), dtype=np.int64)
        self.labels = torch.tensor(source.get_int_targets(), dtype=torch.long)
        self.rows = frame[["cancer_type"]].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return torch.from_numpy(self.features[int(idx)].copy()), int(
            self.labels[idx].item()
        )


def build_specialized_model(
    method: str,
    args: argparse.Namespace,
    dataset_train: TCGAUTDatasetImbalanced,
    device: torch.device,
    tuning_params: dict[str, float],
) -> nn.Module:
    """Train a specialized patch-feature model on preloaded patch features."""
    train_set = PreloadedPatchFeatureDataset(dataset_train)
    settings = training_settings(args)
    n_classes = dataset_train.get_n_classes()
    if method == "patch_feature_cfal":
        return train_cfal_model(
            train_set, n_classes, settings, device, args.seed, tuning_params
        )
    if method == "patch_feature_divide_conquer":
        model, _ = train_divide_conquer_model(
            train_set,
            sorted(dataset_train.dataset["cancer_type"].unique().tolist()),
            n_classes,
            settings,
            device,
            args.seed,
            tuning_params,
        )
        return model
    if method == "patch_feature_oko":
        return train_oko_model(
            train_set, n_classes, settings, device, args.seed, tuning_params
        )
    raise ValueError(f"Unsupported specialized method: {method}")


def training_settings(args: argparse.Namespace) -> dict[str, object]:
    """Return shared trainer settings for specialized patch-feature methods."""
    hidden = args.n_nodes_per_layer[0] if args.n_nodes_per_layer else 512
    return {
        "batch_size": args.batch_size,
        "epochs": args.n_epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "hidden_dim": hidden,
        "dropout": args.dropout,
        "focal_gamma": args.gamma,
        "cfal_lambda": args.cfal_lambda,
        "cfal_sigma": args.cfal_sigma,
        "cfal_gamma": args.cfal_gamma,
        "cfal_beta": args.cfal_beta,
        "dnc_k_clusters": args.dnc_k_clusters,
        "dnc_zscore_bins": args.dnc_zscore_bins,
        "dnc_expert_epochs": args.dnc_expert_epochs,
        "oko_k": args.oko_k,
    }
