"""Shared fixed-step, weighted-sampling training loop for stage one and stage two."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch

from mitigation import TrainHParams
from mitigation.methods import draw_indices

__all__ = ["run_training_steps"]


def run_training_steps(
    weights: np.ndarray,
    hparams: TrainHParams,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    generator: torch.Generator,
    step_fn: Callable[[torch.Tensor], torch.Tensor],
) -> None:
    """Run ``hparams.steps`` optimizer updates, each on a weighted with-replacement index batch.

    ``hparams.micro_batch`` accumulates each update's gradient over chunks of at most that
    many rows (same update as the whole batch, since every loss is a batch mean); ``None``
    forwards the whole batch at once.
    """
    batch_size = hparams.batch_size
    for _ in range(hparams.steps):
        idx = draw_indices(weights, batch_size, generator)
        optimizer.zero_grad(set_to_none=True)
        for chunk in idx.split(hparams.micro_batch or batch_size):
            (step_fn(chunk) * (len(chunk) / batch_size)).backward()
        optimizer.step()
        scheduler.step()
