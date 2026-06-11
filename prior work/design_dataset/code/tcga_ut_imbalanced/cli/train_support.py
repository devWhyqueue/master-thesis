import argparse
import json
import logging
import os
import time
from typing import cast

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tcga_ut_imbalanced.data.dataset import TCGAUTDatasetImbalanced
from tcga_ut_imbalanced.evaluation.tuning_params import parse_tuning_params
from tcga_ut_imbalanced.losses.factory import LossFactory
from tcga_ut_imbalanced.models.mlp import MLP
from tcga_ut_imbalanced.plotting.plots import plot_extended_confusion_matrix
from tcga_ut_imbalanced.training.batch_sampler import BatchBalancingSampler
from tcga_ut_imbalanced.training.loops import train
from tcga_ut_imbalanced.training.patch_feature_adapter import build_specialized_model

logger = logging.getLogger(__name__)


def make_dataloader(
    dataset: TCGAUTDatasetImbalanced,
    batch_size: int,
    sampler: BatchBalancingSampler | None = None,
    shuffle: bool = False,
) -> DataLoader:
    """Create a standard torch dataloader."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle,
        drop_last=False,
    )


def build_mlp(
    args: argparse.Namespace,
    dataset_train: TCGAUTDatasetImbalanced,
    device: torch.device,
    dl_train: DataLoader,
    dl_val: DataLoader | None,
) -> nn.Module:
    """Build and train an MLP model."""
    tuning_params = _load_tuning_params(args)
    if args.training_method in {"cfal", "divide_conquer"}:
        return build_specialized_model(
            args.training_method, args, dataset_train, device, tuning_params
        )
    model = MLP(
        dataset_train.get_feature_dim(),
        args.n_nodes_per_layer,
        dataset_train.get_n_classes(),
        dropout=float(args.dropout),
    )
    model.to(device)
    criterion = _criterion(args, dataset_train, tuning_params)
    optimizer = _optimizer(args, model)
    _train_mlp(args, model, criterion, optimizer, dl_train, dl_val)
    return model


def validation_class_order(
    args: argparse.Namespace,
    dataset_train: TCGAUTDatasetImbalanced,
    dataset_val: TCGAUTDatasetImbalanced,
) -> np.ndarray | list[int]:
    """Return the validation class order."""
    if args.class_names_path is None:
        return np.argsort(dataset_train.get_class_sizes())
    with open(args.class_names_path) as file:
        class_names = json.load(file)
    return [
        dataset_val.features_str_to_int_map[class_name] for class_name in class_names
    ]


def save_validation_plot(
    base_path: str,
    dataset_val: TCGAUTDatasetImbalanced,
    class_order: np.ndarray | list[int],
    result: dict[str, object],
) -> None:
    """Save the extended confusion matrix validation plot."""
    viz_path = os.path.join(base_path, "visualizations")
    os.makedirs(viz_path, exist_ok=True)
    class_names = [dataset_val.get_int_to_class_map()[index] for index in class_order]
    fig, _ = plot_extended_confusion_matrix(
        cast(np.ndarray, result["confusion_matrix"]),
        cast(np.ndarray, result["recall_per_class"]),
        cast(np.ndarray, result["precision_per_class"]),
        class_names=class_names,
    )
    fig.savefig(
        os.path.join(viz_path, "extended_confusion_matrix.png"),
        dpi=300,
        bbox_inches="tight",
    )


def training_base_path(args: argparse.Namespace) -> str:
    """Return the output directory for one training run."""
    timestamp = time.time_ns() // 1_000_000
    return (
        os.path.join(args.results_save_path, str(timestamp))
        if args.store_timestamp
        else args.results_save_path
    )


def _optimizer(args: argparse.Namespace, model: MLP) -> torch.optim.Optimizer:
    if args.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
    return torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )


def _load_tuning_params(args: argparse.Namespace) -> dict[str, float]:
    return parse_tuning_params("patch", args.training_method, args.tuning_params)


def _criterion(
    args: argparse.Namespace,
    dataset_train: TCGAUTDatasetImbalanced,
    tuning_params: dict[str, float],
) -> nn.Module:
    gamma = tuning_params.get("focal_gamma", args.gamma)
    metric_weight = float(
        tuning_params.get("metric_loss_weight", args.metric_loss_weight) or 0.0
    )
    weight_power = float(tuning_params.get("weight_power", 1.0))
    return LossFactory.build(
        args.loss,
        gamma,
        args.alpha,
        dataset_train.get_n_classes(),
        dataset_train.get_class_sizes(),
        metric_weight,
        weight_power,
    )


def _train_mlp(
    args: argparse.Namespace,
    model: MLP,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    dl_train: DataLoader,
    dl_val: DataLoader | None,
) -> None:
    train(model, dl_train, args.n_epochs, criterion, optimizer, dl_val)
