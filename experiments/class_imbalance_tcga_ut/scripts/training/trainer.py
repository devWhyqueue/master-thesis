from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import torch
from torch import nn

from scripts.training.oko import _train_oko
from scripts.training.split import _slice_split_rows
from scripts.training.support import (
    FeatureDataset,
    Mlp,
    _batch_center_loss,
    _build_criterion,
    _evaluate,
    _interpolate_minority,
    _make_loader,
    _resolve_device,
)


@dataclass
class SplitSets:
    train_frame: pd.DataFrame
    train_dataset: FeatureDataset
    val_dataset: FeatureDataset
    test_dataset: FeatureDataset


def _load_split(
    paths: dict[str, Path], seed: int, smoke: bool, config: dict
) -> pd.DataFrame:
    frame = pd.read_csv(paths["data"] / f"manifest_splits_seed={seed}.csv")
    max_train = config["training"].get("max_train_rows")
    max_eval = config["training"].get("max_eval_rows")
    if smoke:
        max_train = min(int(max_train or 8), 8)
        max_eval = min(int(max_eval or 4), 4)
    return _slice_split_rows(frame, max_train, max_eval)


def _train_mlp(
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
    class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
    split_sets = _split_datasets(frame, class_to_idx)
    train_labels = (
        split_sets.train_frame["cancer_type"].replace(class_to_idx).to_numpy()
    )
    model = _build_model(split_sets.train_dataset, class_names, training, device)
    optimizer = _build_optimizer(model, training)
    if method == "oko":
        _run_oko_training(
            model,
            split_sets.train_dataset,
            train_labels,
            training,
            optimizer,
            device,
            seed,
        )
    else:
        _run_supervised_training(
            model,
            split_sets.train_dataset,
            train_labels,
            method,
            training,
            optimizer,
            device,
            seed,
        )
    return _save_and_evaluate(model, split_sets, class_names, device, result_dir)


def _split_datasets(frame: pd.DataFrame, class_to_idx: dict[str, int]) -> SplitSets:
    train_frame = cast(pd.DataFrame, frame[frame["split"] == "train"])
    return SplitSets(
        train_frame=train_frame,
        train_dataset=FeatureDataset(train_frame, class_to_idx),
        val_dataset=FeatureDataset(
            cast(pd.DataFrame, frame[frame["split"] == "val"]), class_to_idx
        ),
        test_dataset=FeatureDataset(
            cast(pd.DataFrame, frame[frame["split"] == "test"]), class_to_idx
        ),
    )


def _build_model(
    train_dataset: FeatureDataset,
    class_names: list[str],
    training: dict,
    device: torch.device,
) -> nn.Module:
    sample_feature, _ = train_dataset[0]
    model = Mlp(
        input_dim=int(sample_feature.numel()),
        hidden_dims=list(training["hidden_dims"]),
        output_dim=len(class_names),
        dropout=float(training["dropout"]),
    )
    return model.to(device)


def _build_optimizer(model: nn.Module, training: dict) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )


def _run_oko_training(
    model: nn.Module,
    train_dataset: FeatureDataset,
    train_labels: np.ndarray,
    training: dict,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    seed: int,
) -> None:
    _train_oko(
        model=model,
        dataset=train_dataset,
        labels=train_labels,
        epochs=int(training["epochs"]),
        steps_per_epoch=max(1, len(train_dataset) // int(training["batch_size"])),
        k_oko=int(training["oko_k"]),
        optimizer=optimizer,
        device=device,
        seed=seed,
    )


def _run_supervised_training(
    model: nn.Module,
    train_dataset: FeatureDataset,
    train_labels: np.ndarray,
    method: str,
    training: dict,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    seed: int,
) -> None:
    loader = _make_loader(
        train_dataset, train_labels, method, int(training["batch_size"]), seed
    )
    criterion = _build_criterion(
        method,
        train_labels,
        int(len(np.unique(train_labels))),
        float(training["focal_gamma"]),
        device,
    )
    center_weight = (
        float(training["center_loss_weight"]) if method == "center_ce" else 0.0
    )
    for _ in range(int(training["epochs"])):
        _run_supervised_epoch(
            model,
            loader,
            train_labels,
            method,
            criterion,
            optimizer,
            device,
            center_weight,
        )


def _run_supervised_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    train_labels: np.ndarray,
    method: str,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    center_weight: float,
) -> None:
    model.train()
    for features, targets in loader:
        features = features.to(device)
        targets = targets.to(device)
        if method == "feature_interpolation_ce":
            features, targets = _interpolate_minority(features, targets, train_labels)
        logits = model.forward(features)
        loss = criterion.forward(logits, targets)
        if center_weight:
            loss = loss + center_weight * _batch_center_loss(logits, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def _save_and_evaluate(
    model: nn.Module,
    split_sets: SplitSets,
    class_names: list[str],
    device: torch.device,
    result_dir: Path,
) -> dict[str, dict[str, object]]:
    val_results = _evaluate(model, split_sets.val_dataset, class_names, device)
    test_results = _evaluate(model, split_sets.test_dataset, class_names, device)
    torch.save(model.state_dict(), result_dir / "model.pt")
    return {"val": val_results, "test": test_results}
