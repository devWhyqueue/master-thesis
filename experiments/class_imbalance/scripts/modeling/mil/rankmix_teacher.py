from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from torch import nn

from scripts.modeling.mil.bag.dataset import AttentionMil, BagFeatureDataset
from scripts.modeling.training.split import _progress_payload, _write_training_progress


class ModelBuilder(Protocol):
    """Construct an attention MIL model for a dataset."""

    def __call__(
        self,
        dataset: BagFeatureDataset,
        class_names: list[str],
        training: dict,
        device: torch.device,
    ) -> AttentionMil: ...


class LoaderFactory(Protocol):
    """Construct a bag DataLoader for a training method."""

    def __call__(
        self,
        dataset: BagFeatureDataset,
        labels: np.ndarray,
        method: str,
        batch_size: int,
        seed: int,
    ) -> torch.utils.data.DataLoader: ...


def load_rankmix_teacher(
    result_dir: Path,
    build_model: ModelBuilder,
    train_dataset: BagFeatureDataset,
    class_names: list[str],
    training: dict,
    device: torch.device,
    seed: int,
) -> AttentionMil | None:
    """Reuse the matched plain MIL checkpoint as RankMix's stage-one teacher."""
    checkpoint = result_dir.parents[1] / "mil_ce" / f"seed={seed}" / "model.pt"
    if not checkpoint.exists():
        return None
    teacher = build_model(train_dataset, class_names, training, device)
    teacher.load_state_dict(torch.load(checkpoint, map_location=device))
    return _freeze_teacher(teacher)


def train_rankmix_teacher(
    model: AttentionMil,
    dataset: BagFeatureDataset,
    labels: np.ndarray,
    training: dict,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    seed: int,
    result_dir: Path,
    loader_factory: LoaderFactory,
) -> AttentionMil:
    """Train the stage-one general MIL teacher used by RankMix."""
    loader = loader_factory(
        dataset, labels, "mil_ce", int(training["bag_batch_size"]), seed
    )
    epochs = int(training["rankmix_teacher_epochs"])
    for epoch in range(1, epochs + 1):
        loss = _run_teacher_epoch(model, loader, optimizer, device)
        _write_training_progress(
            result_dir,
            _progress_payload(
                "rankmix_teacher", seed, device, "running", epoch, epochs
            ),
            loss,
        )
    return _freeze_teacher(deepcopy(model))


def _run_teacher_epoch(
    model: AttentionMil,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    losses: list[float] = []
    for bags, targets in loader:
        logits, _, _ = model.forward_bags([bag.to(device) for bag in bags])
        loss = nn.functional.cross_entropy(logits, targets.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
    return float(np.mean(losses))


def _freeze_teacher(teacher: AttentionMil) -> AttentionMil:
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher
