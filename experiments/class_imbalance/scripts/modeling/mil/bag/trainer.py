from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from scripts.common import output_root, write_json
from scripts.modeling.mil.bag.losses import bag_loss
from scripts.modeling.mil.bag.dataset import (
    AttentionMil,
    BagFeatureDataset,
    DualExpertMil,
    bag_collate,
    class_weights,
    infer_input_dim,
)
from scripts.modeling.mil.rankmix_teacher import (
    load_rankmix_teacher,
    train_rankmix_teacher,
)
from scripts.modeling.training.eval import _save_and_evaluate_bags
from scripts.modeling.training.split import _progress_payload, _write_training_progress
from scripts.modeling.training.support import _resolve_device

BALANCED_SAMPLER_METHODS = frozenset(
    {"mil_balanced_sampler_ce", "sc_mil", "rankmix_mil"}
)


def _train_bag_method(
    method: str,
    frame: pd.DataFrame,
    class_names: list[str],
    config: dict,
    seed: int,
    result_dir: Path,
    smoke: bool = False,
) -> dict[str, dict[str, object]]:
    """Train one WSI-bag benchmark method."""
    torch.manual_seed(seed)
    training = config["wsi_training"]
    device = _resolve_device(training["device"])
    train_dataset, val_dataset, test_dataset = _split_bag_datasets(
        frame, class_names, training, config, seed, smoke
    )
    labels = train_dataset.labels.cpu().numpy()
    build_model = _build_mde_model if method == "mde_mil" else _build_model
    model = build_model(train_dataset, class_names, training, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    teacher = None
    if method == "rankmix_mil":
        teacher = load_rankmix_teacher(
            result_dir,
            _build_model,
            train_dataset,
            class_names,
            training,
            device,
            seed,
        )
        if teacher is None:
            teacher = train_rankmix_teacher(
                model,
                train_dataset,
                labels,
                training,
                optimizer,
                device,
                seed,
                result_dir,
                _loader,
            )
        model = _build_model(train_dataset, class_names, training, device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
        )
    diagnostics = _run_training(
        method,
        model,
        train_dataset,
        labels,
        training,
        optimizer,
        device,
        seed,
        result_dir,
        teacher,
    )
    write_json(result_dir / "activation_diagnostics.json", diagnostics)
    return _save_and_evaluate_bags(
        model, val_dataset, test_dataset, class_names, device, result_dir
    )


def _split_bag_datasets(
    frame: pd.DataFrame,
    class_names: list[str],
    training: dict,
    config: dict,
    seed: int,
    smoke: bool = False,
) -> tuple[BagFeatureDataset, BagFeatureDataset, BagFeatureDataset]:
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    max_instances = training.get("max_instances_per_bag")
    cache_dir = _cache_dir(config, seed)
    return tuple(
        BagFeatureDataset(
            cast(pd.DataFrame, frame[frame["split"] == split]),
            class_to_idx,
            max_instances,
            cache_dir if _cache_exists(cache_dir, split) and not smoke else None,
            split,
        )
        for split in ["train", "val", "test"]
    )  # type: ignore[return-value]


def _cache_dir(config: dict, seed: int) -> Path:
    return output_root(config) / "data" / "wsi_bag_cache" / f"seed={seed}"


def _cache_exists(cache_dir: Path, split: str) -> bool:
    return (cache_dir / f"{split}_features.npy").exists() and (
        cache_dir / f"{split}_offsets.npy"
    ).exists()


def _build_model(
    dataset: BagFeatureDataset,
    class_names: list[str],
    training: dict,
    device: torch.device,
) -> AttentionMil:
    model = AttentionMil(
        infer_input_dim(dataset),
        int(training["hidden_dim"]),
        len(class_names),
        float(training["dropout"]),
    )
    return cast(AttentionMil, model.to(device))


def _build_mde_model(
    dataset: BagFeatureDataset,
    class_names: list[str],
    training: dict,
    device: torch.device,
) -> DualExpertMil:
    model = DualExpertMil(
        infer_input_dim(dataset),
        int(training["hidden_dim"]),
        len(class_names),
        float(training["dropout"]),
    )
    return cast(DualExpertMil, model.to(device))


def _loader(
    dataset: BagFeatureDataset,
    labels: np.ndarray,
    method: str,
    batch_size: int,
    seed: int,
    sampler_power: float = 1.0,
    *,
    balanced: bool | None = None,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    use_balanced = method in BALANCED_SAMPLER_METHODS if balanced is None else balanced
    if not use_balanced:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
            collate_fn=bag_collate,
        )
    counts = np.bincount(labels)
    sample_weights = [
        float((1.0 / counts[int(label)]) ** sampler_power) for label in labels
    ]
    sampler = WeightedRandomSampler(sample_weights, len(labels), True, generator)
    return DataLoader(
        dataset, batch_size=batch_size, sampler=sampler, collate_fn=bag_collate
    )


def _run_training(
    method: str,
    model: nn.Module,
    dataset: BagFeatureDataset,
    labels: np.ndarray,
    training: dict,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    seed: int,
    result_dir: Path,
    teacher: AttentionMil | None = None,
) -> dict[str, int]:
    if method == "mde_mil":
        return _run_mde_training(
            model,
            dataset,
            labels,
            training,
            optimizer,
            device,
            seed,
            result_dir,
        )
    loader = _loader(
        dataset,
        labels,
        method,
        int(training["bag_batch_size"]),
        seed,
        float(training.get("sampler_power", 1.0)),
    )
    weights = class_weights(
        labels,
        int(len(np.unique(labels))),
        power=float(training.get("weight_power", 1.0)),
    ).to(device)
    totals = {"mixed_examples": 0, "positive_pairs": 0}
    epochs = int(training["epochs"])
    total_steps = max(1, epochs * len(loader))
    step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        for bags, targets in loader:
            step += 1
            loss, diagnostics = bag_loss(
                method,
                model,
                [bag.to(device) for bag in bags],
                targets.to(device),
                weights,
                step / total_steps,
                training,
                teacher,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
            for key, value in diagnostics.items():
                totals[key] = totals.get(key, 0) + int(value)
        _write_training_progress(
            result_dir,
            _progress_payload(method, seed, device, "running", epoch, epochs),
            float(np.mean(losses)),
        )
    return totals


def _run_mde_training(
    model: nn.Module,
    dataset: BagFeatureDataset,
    labels: np.ndarray,
    training: dict,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    seed: int,
    result_dir: Path,
) -> dict[str, int]:
    batch_size = int(training["bag_batch_size"])
    sampler_power = float(training.get("sampler_power", 1.0))
    loader_u = _loader(
        dataset, labels, "mil_ce", batch_size, seed, sampler_power, balanced=False
    )
    loader_b = _loader(
        dataset,
        labels,
        "mil_balanced_sampler_ce",
        batch_size,
        seed,
        sampler_power,
        balanced=True,
    )
    weights = class_weights(
        labels,
        int(len(np.unique(labels))),
        power=float(training.get("weight_power", 1.0)),
    ).to(device)
    totals = {"branch_u_batches": 0, "branch_b_batches": 0}
    epochs = int(training["epochs"])
    total_steps = max(1, epochs * len(loader_u))
    step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        for (bags_u, targets_u), (bags_b, targets_b) in zip(
            loader_u, loader_b, strict=True
        ):
            step += 1
            loss, diagnostics = bag_loss(
                "mde_mil",
                model,
                [bag.to(device) for bag in bags_u],
                targets_u.to(device),
                weights,
                step / total_steps,
                training,
                bags_b=[bag.to(device) for bag in bags_b],
                targets_b=targets_b.to(device),
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
            for key, value in diagnostics.items():
                totals[key] = totals.get(key, 0) + int(value)
        _write_training_progress(
            result_dir,
            _progress_payload("mde_mil", seed, device, "running", epoch, epochs),
            float(np.mean(losses)),
        )
    return totals
