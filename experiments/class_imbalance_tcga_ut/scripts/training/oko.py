from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from scripts.training.support import FeatureDataset


@dataclass
class OkoPayload:
    pair_class: int
    indices: list[int]


def _train_oko(
    model: nn.Module,
    dataset: FeatureDataset,
    labels: np.ndarray,
    epochs: int,
    steps_per_epoch: int,
    k_oko: int,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    seed: int,
    progress_callback: Callable[[int], None] | None = None,
) -> None:
    rng = np.random.default_rng(seed)
    class_indices = [
        np.where(labels == class_id)[0] for class_id in range(int(labels.max()) + 1)
    ]
    valid_pair_classes = [
        class_id for class_id, idxs in enumerate(class_indices) if len(idxs) >= 2
    ]
    if not valid_pair_classes:
        raise RuntimeError(
            "OKO requires at least one class with two training examples."
        )
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, epochs + 1):
        _run_oko_epoch(
            model,
            dataset,
            class_indices,
            valid_pair_classes,
            k_oko,
            optimizer,
            device,
            rng,
            criterion,
            steps_per_epoch,
        )
        if progress_callback is not None:
            progress_callback(epoch)


def _run_oko_epoch(
    model: nn.Module,
    dataset: FeatureDataset,
    class_indices: list[np.ndarray],
    valid_pair_classes: list[int],
    k_oko: int,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    rng: np.random.Generator,
    criterion: nn.Module,
    steps_per_epoch: int,
) -> None:
    model.train()
    for _ in range(steps_per_epoch):
        payload = _oko_step_payload(class_indices, valid_pair_classes, k_oko, rng)
        if payload is None:
            continue
        features = torch.stack([dataset[index][0] for index in payload.indices]).to(
            device
        )
        target = torch.tensor([payload.pair_class], device=device)
        logits = model(features).sum(dim=0, keepdim=True)
        loss = criterion(logits, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def _oko_step_payload(
    class_indices: list[np.ndarray],
    valid_pair_classes: list[int],
    k_oko: int,
    rng: np.random.Generator,
) -> OkoPayload | None:
    pair_class = int(rng.choice(valid_pair_classes))
    odd_classes = [
        cid
        for cid, idxs in enumerate(class_indices)
        if cid != pair_class and len(idxs) > 0
    ]
    if len(odd_classes) < k_oko:
        return None
    pair_indices = rng.choice(class_indices[pair_class], size=2, replace=False)
    selected_odd = rng.choice(odd_classes, size=k_oko, replace=False)
    odd_indices = [
        int(rng.choice(class_indices[class_id])) for class_id in selected_odd
    ]
    return OkoPayload(
        pair_class=pair_class,
        indices=list(map(int, pair_indices)) + odd_indices,
    )
