import logging
from typing import cast

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from analysis.evaluation.metrics import test_model
from modeling.models.mlp import MLP

logger = logging.getLogger(__name__)


def train(
    model: MLP,
    dataloader: DataLoader,
    n_epochs: int,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    dl_val: DataLoader | None = None,
) -> None:
    """Train a model for multiple single-example epochs."""
    logger.info("Beginning training")
    model.train()
    for epoch in range(1, n_epochs + 1):
        train_single_epoch(model, dataloader, criterion, optimizer, epoch, dl_val)


def train_single_epoch(
    model: MLP,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    dl_val: DataLoader | None = None,
) -> None:
    """Train a model for one single-example epoch."""
    logger.info("Beginning epoch %s", epoch)
    correct = 0
    seen = 0
    for batch in dataloader:
        correct, seen = _train_batch(model, criterion, optimizer, batch, correct, seen)
    logger.info(
        "Epoch %s finished, samples=%s, training accuracy=%.5f",
        epoch,
        seen,
        correct / seen if seen else 0.0,
    )
    if dl_val is not None:
        _log_validation(model, dl_val)


def _train_batch(
    model: MLP,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    correct: int,
    seen: int,
) -> tuple[int, int]:
    optimizer.zero_grad()
    first_layer = cast(nn.Linear, model.model[0])
    output = model(batch["features"].to(first_layer.weight.dtype))
    output = output.unsqueeze(0) if output.ndim == 1 else output
    targets = batch["target"].to(output.device)
    loss = criterion(output, targets)
    loss.backward()
    optimizer.step()
    predictions = torch.argmax(output, dim=-1).detach().cpu().numpy()
    labels = targets.detach().cpu().numpy()
    return correct + int(np.sum(predictions == labels)), seen + len(batch["target"])


def _log_validation(model: MLP, dl_val: DataLoader) -> None:
    result = test_model(model, dl_val)
    logger.info(
        "Validation Accuracy = %.5f, Validation Balanced Accuracy = %.5f",
        result["accuracy"],
        result["balanced_accuracy"],
    )
