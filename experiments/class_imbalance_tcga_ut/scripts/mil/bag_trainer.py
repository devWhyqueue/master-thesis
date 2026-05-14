from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from scripts.mil.bags import (
    AttentionMil,
    BagFeatureDataset,
    MdeMil,
    SyntheticBagFeatureDataset,
    bag_collate,
    class_weights,
    infer_input_dim,
)
from scripts.training.eval import _save_and_evaluate_bags
from scripts.mil.bag_losses import bag_loss
from scripts.mil.synthetic_features import append_encoded_gan_features
from scripts.training.split import _progress_payload, _write_training_progress
from scripts.training.support import _resolve_device


def _train_bag_method(
    method: str,
    frame: pd.DataFrame,
    class_names: list[str],
    config: dict,
    seed: int,
    result_dir: Path,
) -> dict[str, dict[str, object]]:
    torch.manual_seed(seed)
    training = config["training"]
    device = _resolve_device(training["device"])
    train_dataset, val_dataset, test_dataset, feature_gan_noise = _split_bag_datasets(
        frame, class_names, config, method, seed, result_dir
    )
    train_labels = train_dataset.labels.cpu().numpy()
    model = _build_bag_model(method, train_dataset, class_names, training, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    _run_bag_training(
        method,
        model,
        train_dataset,
        train_labels,
        training,
        optimizer,
        device,
        seed,
        result_dir,
        feature_gan_noise,
    )
    return _save_and_evaluate_bags(
        model, val_dataset, test_dataset, class_names, device, result_dir
    )


def _class_map(class_names: list[str]) -> dict[str, int]:
    return {class_name: idx for idx, class_name in enumerate(class_names)}


def _split_bag_datasets(
    frame: pd.DataFrame,
    class_names: list[str],
    config: dict,
    method: str,
    seed: int,
    result_dir: Path,
) -> tuple[SyntheticBagFeatureDataset, BagFeatureDataset, BagFeatureDataset, bool]:
    class_to_idx = _class_map(class_names)
    training = config["training"]
    max_instances = training.get("max_instances_per_bag")
    train_frame = cast(pd.DataFrame, frame[frame["split"] == "train"])
    train_dataset = SyntheticBagFeatureDataset(train_frame, class_to_idx, max_instances)
    feature_gan_noise = not append_encoded_gan_features(
        train_dataset,
        class_to_idx,
        max_instances,
        config,
        method,
        seed,
        result_dir,
    )
    return (
        train_dataset,
        BagFeatureDataset(
            cast(pd.DataFrame, frame[frame["split"] == "val"]),
            class_to_idx,
            max_instances,
        ),
        BagFeatureDataset(
            cast(pd.DataFrame, frame[frame["split"] == "test"]),
            class_to_idx,
            max_instances,
        ),
        feature_gan_noise,
    )


def _build_bag_model(
    method: str,
    train_dataset: BagFeatureDataset,
    class_names: list[str],
    training: dict,
    device: torch.device,
) -> nn.Module:
    hidden_dims = list(training["hidden_dims"])
    hidden_dim = int(hidden_dims[0] if hidden_dims else 256)
    input_dim = infer_input_dim(train_dataset)
    if method == "mde_mil":
        model: nn.Module = MdeMil(
            input_dim, hidden_dim, len(class_names), float(training["dropout"])
        )
    else:
        model = AttentionMil(
            input_dim, hidden_dim, len(class_names), float(training["dropout"])
        )
    return model.to(device)


def _run_bag_training(
    method: str,
    model: nn.Module,
    dataset: BagFeatureDataset,
    labels: np.ndarray,
    training: dict,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    seed: int,
    result_dir: Path,
    feature_gan_noise: bool,
) -> None:
    loader = _bag_loader(dataset, labels, method, int(training["bag_batch_size"]), seed)
    weights = class_weights(labels, int(len(np.unique(labels)))).to(device)
    epochs = int(training["epochs"])
    for epoch in range(1, epochs + 1):
        loss = _run_bag_epoch(
            method,
            model,
            loader,
            labels,
            weights,
            optimizer,
            device,
            feature_gan_noise,
        )
        _write_training_progress(
            result_dir,
            _progress_payload(method, seed, device, "running", epoch, epochs),
            loss,
        )


def _bag_loader(
    dataset: BagFeatureDataset,
    labels: np.ndarray,
    method: str,
    batch_size: int,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    if method == "mde_mil":
        counts = np.bincount(labels)
        sample_weights = [float(1.0 / counts[int(label)]) for label in labels]
        sampler = WeightedRandomSampler(sample_weights, len(labels), True, generator)
        return DataLoader(
            dataset, batch_size=batch_size, sampler=sampler, collate_fn=bag_collate
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=bag_collate,
    )


def _run_bag_epoch(
    method: str,
    model: nn.Module,
    loader: DataLoader,
    labels: np.ndarray,
    weights: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    feature_gan_noise: bool,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    for bags, targets in loader:
        targets = targets.to(device)
        device_bags = [bag.to(device) for bag in bags]
        loss = bag_loss(
            method,
            model,
            device_bags,
            targets,
            labels,
            weights,
            feature_gan_noise,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu().item())
        n_batches += 1
    return total_loss / max(n_batches, 1)
