import logging
from collections.abc import Sequence
from typing import cast

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.dataset import TCGAUTDatasetImbalanced
from analysis.evaluation.metrics import test_model
from modeling.models.mlp import MLP

logger = logging.getLogger(__name__)


def train_oko(
    model: MLP,
    dataset: TCGAUTDatasetImbalanced,
    n_epochs: int,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    dl_val: DataLoader | None = None,
    k_oko: int = 1,
    seed: int = 0,
) -> None:
    """Train with the odd-k-out set objective."""
    logger.info("Beginning OKO training")
    model.train()
    rng = np.random.default_rng(seed)
    for epoch in range(1, n_epochs + 1):
        train_single_epoch_oko(model, dataset, criterion, optimizer, epoch, k_oko, rng)
        if dl_val is not None:
            _log_validation(model, dl_val)


def train_single_epoch_oko(
    model: MLP,
    dataset: TCGAUTDatasetImbalanced,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    k_oko: int,
    rng: np.random.Generator,
) -> None:
    """Train one odd-k-out epoch."""
    logger.info("Beginning OKO epoch %s", epoch)
    class_indices = _class_indices(dataset)
    stats = _OkoStats()
    for _ in range(len(dataset)):
        stats = _train_oko_set(
            model, dataset, criterion, optimizer, class_indices, k_oko, rng, stats
        )
        if stats.n_sets > 0 and stats.n_sets % 50 == 0:
            _log_oko_progress(epoch, stats)
    _log_oko_epoch_end(epoch, stats)


def sample_oko_set_indices(
    n_classes: int,
    class_indices: Sequence[np.ndarray],
    k_oko: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray | None, int | None]:
    """Sample indices for one odd-k-out set."""
    pair_classes = [
        class_index
        for class_index in range(n_classes)
        if len(class_indices[class_index]) >= 2
    ]
    if not pair_classes:
        raise RuntimeError(
            "No class has at least two examples; cannot construct OKO sets."
        )
    pair_class = int(rng.choice(pair_classes))
    odd_classes = _sample_odd_classes(n_classes, class_indices, pair_class, k_oko, rng)
    if odd_classes is None:
        return None, None
    pair_indices = rng.choice(class_indices[pair_class], size=2, replace=False)
    odd_indices = [
        rng.choice(class_indices[class_index]) for class_index in odd_classes
    ]
    return np.concatenate(
        [pair_indices, np.array(odd_indices, dtype=np.int64)]
    ), pair_class


def build_oko_batch(
    dataset: TCGAUTDatasetImbalanced,
    idxs: Sequence[int],
    y_pair: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build an odd-k-out feature set and target tensor."""
    features = [
        cast(torch.Tensor, dataset[int(index)]["features"])
        .to(device=device, dtype=dtype)
        .unsqueeze(0)
        for index in idxs
    ]
    return torch.cat(features, dim=0), torch.tensor([y_pair], device=device)


class _OkoStats:
    def __init__(self, right_preds: int = 0, n_sets: int = 0) -> None:
        self.right_preds = right_preds
        self.n_sets = n_sets


def _class_indices(dataset: TCGAUTDatasetImbalanced) -> list[np.ndarray]:
    int_targets = dataset.get_int_targets()
    return [
        np.where(int_targets == class_index)[0]
        for class_index in range(dataset.get_n_classes())
    ]


def _train_oko_set(
    model: MLP,
    dataset: TCGAUTDatasetImbalanced,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    class_indices: Sequence[np.ndarray],
    k_oko: int,
    rng: np.random.Generator,
    stats: _OkoStats,
) -> _OkoStats:
    indices, target_class = sample_oko_set_indices(
        dataset.get_n_classes(), class_indices, k_oko, rng
    )
    if indices is None or target_class is None:
        return stats
    set_logits, target = _forward_oko_set(
        model, dataset, indices.tolist(), target_class
    )
    _optimize_oko_set(criterion, optimizer, set_logits, target)
    return _OkoStats(
        stats.right_preds
        + int((torch.argmax(set_logits, dim=-1) == target).sum().item()),
        stats.n_sets + 1,
    )


def _forward_oko_set(
    model: MLP,
    dataset: TCGAUTDatasetImbalanced,
    indices: Sequence[int],
    target_class: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = next(model.parameters()).device
    first_layer = cast(nn.Linear, model.model[0])
    features, target = build_oko_batch(
        dataset, indices, target_class, device, first_layer.weight.dtype
    )
    logits = model(features).squeeze()
    logits = logits.unsqueeze(0) if logits.ndim == 1 else logits
    return logits.sum(dim=0, keepdim=True), target


def _optimize_oko_set(
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    set_logits: torch.Tensor,
    target: torch.Tensor,
) -> None:
    optimizer.zero_grad()
    loss = criterion(set_logits, target)
    loss.backward()
    optimizer.step()


def _sample_odd_classes(
    n_classes: int,
    class_indices: Sequence[np.ndarray],
    pair_class: int,
    k_oko: int,
    rng: np.random.Generator,
) -> np.ndarray | None:
    available = [
        class_index
        for class_index in range(n_classes)
        if class_index != pair_class and len(class_indices[class_index]) > 0
    ]
    return (
        None
        if len(available) < k_oko
        else rng.choice(available, size=k_oko, replace=False)
    )


def _log_validation(model: MLP, dl_val: DataLoader) -> None:
    result = test_model(model, dl_val)
    logger.info(
        "Validation Accuracy = %.5f, Validation Balanced Accuracy = %.5f",
        result["accuracy"],
        result["balanced_accuracy"],
    )


def _log_oko_progress(epoch: int, stats: _OkoStats) -> None:
    logger.info(
        "OKO Epoch %s, Sets encountered %s, Training Accuracy (pair class) = %.5f",
        epoch,
        stats.n_sets,
        stats.right_preds / stats.n_sets,
    )


def _log_oko_epoch_end(epoch: int, stats: _OkoStats) -> None:
    if stats.n_sets == 0:
        logger.info("Finished OKO epoch %s, no valid sets were constructed.", epoch)
        return
    logger.info(
        "Finished OKO epoch %s, Training Accuracy (pair class) = %.5f",
        epoch,
        stats.right_preds / stats.n_sets,
    )
