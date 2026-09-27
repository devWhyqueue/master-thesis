"""Shared fixed-step, weighted-sampling training loop for stage one and stage two."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch

from mitigation.methods import draw_indices

__all__ = ["run_training_steps"]


def run_training_steps(
    weights: np.ndarray,
    steps: int,
    batch_size: int,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    generator: torch.Generator,
    step_fn: Callable[[torch.Tensor], torch.Tensor],
) -> None:
    """Run ``steps`` optimizer updates, each on a weighted with-replacement index batch."""
    for _ in range(steps):
        idx = draw_indices(weights, batch_size, generator)
        optimizer.zero_grad(set_to_none=True)
        loss = step_fn(idx)
        loss.backward()
        optimizer.step()
        scheduler.step()
