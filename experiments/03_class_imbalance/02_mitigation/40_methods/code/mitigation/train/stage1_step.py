"""Per-step loss closures for stage-one training: one per method, sharing one encoder forward."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from mitigation.encoder import pool_tokens
from mitigation.images import normalize_batch
from mitigation.methods import gcl_loss, la_loss, mixup_batch, mixup_loss

__all__ = ["StepModel", "build_loss_step"]


class StepModel(NamedTuple):
    """The encoder, head, and resident training images/targets one step forward-passes."""

    encoder: nn.Module
    head: nn.Module
    images: torch.Tensor
    y_t: torch.Tensor
    mean: tuple[float, ...]
    std: tuple[float, ...]


def _method_step_fn(
    method: str,
    param: float,
    head: nn.Module,
    log_prior: torch.Tensor | None,
    delta: torch.Tensor | None,
    gcl_s: float | None,
    generator: torch.Generator,
) -> Any:
    """Build the (features, y) -> loss function for one stage-one method."""
    if method == "la":
        assert log_prior is not None
        return lambda features, y: la_loss(head(features), y, log_prior, param)
    if method == "gcl":
        assert delta is not None and gcl_s is not None
        return lambda features, y: gcl_loss(
            head(features), y, delta, gcl_s, param, generator
        )
    return lambda features, y: F.cross_entropy(head(features), y)


def _build_step_fn(
    method: str,
    param: float,
    model: StepModel,
    device: torch.device,
    method_step: Any,
    generator: torch.Generator,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build the index-batch -> loss closure the training loop calls every step."""

    def step(idx: torch.Tensor) -> torch.Tensor:
        """Forward one sampled index batch through the encoder and head; return its loss."""
        idx = idx.to(device)
        x = normalize_batch(model.images[idx], model.mean, model.std)
        y_batch = model.y_t[idx]
        perm, lam = None, 1.0
        if method == "mixup":
            x, perm, lam = mixup_batch(x, param, generator)
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            features = pool_tokens(model.encoder(x))
            if method == "mixup":
                assert perm is not None
                return mixup_loss(model.head(features), y_batch, perm, lam)
            return method_step(features, y_batch)

    return step


def build_loss_step(
    method: str,
    param: float,
    model: StepModel,
    device: torch.device,
    extras: tuple[torch.Tensor | None, torch.Tensor | None, float | None],
    generator: torch.Generator,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build the full index-batch -> loss closure for one stage-one method."""
    log_prior, delta, gcl_s = extras
    method_step = _method_step_fn(
        method, param, model.head, log_prior, delta, gcl_s, generator
    )
    return _build_step_fn(method, param, model, device, method_step, generator)
