"""Per-step loss closures for stage-one training: one per method, sharing one encoder forward."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from mitigation.encoder import pool_tokens
from mitigation.images import normalize_batch
from mitigation.methods import cuda_batch, gcl_loss, la_loss, update_levels

__all__ = ["StepModel", "build_loss_step"]

logger = logging.getLogger(__name__)


class StepModel(NamedTuple):
    """The encoder, head, and resident training images/targets one step forward-passes."""

    encoder: nn.Module
    head: nn.Module
    images: torch.Tensor
    y_t: torch.Tensor
    mean: tuple[float, ...]
    std: tuple[float, ...]


class _CudaState:
    """Per-run mutable CUDA state: per-class levels of learning, updated once per epoch."""

    def __init__(
        self,
        model: StepModel,
        device: torch.device,
        params: tuple[float, int, int, int],
    ) -> None:
        """``params`` is (gamma, check_size, max_strength, steps_per_epoch) (Eq. cuda-lol)."""
        self._model = model
        self._device = device
        self.gamma, self.check_size, self.max_strength, self._steps_per_epoch = params
        num_classes = int(model.y_t.max().item()) + 1
        self.levels = torch.zeros(num_classes, dtype=torch.long, device=device)
        self.class_rows = [model.images[model.y_t == c] for c in range(num_classes)]
        self._step_count = 0

    def _predict(self, batch_u8: torch.Tensor) -> torch.Tensor:
        """Argmax class prediction for an augmented check batch, no grad (Eq. cuda-lol)."""
        with (
            torch.no_grad(),
            torch.autocast(
                device_type=self._device.type,
                dtype=torch.bfloat16,
                enabled=self._device.type == "cuda",
            ),
        ):
            x = normalize_batch(batch_u8, self._model.mean, self._model.std)
            features = pool_tokens(self._model.encoder(x))
            logits = self._model.head(features)
        return logits.argmax(dim=-1)

    def augment_batch(
        self,
        raw: torch.Tensor,
        y_batch: torch.Tensor,
        p_aug: float,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """Replace each row by its class's level-of-learning augmentation (Eq. cuda-aug)."""
        self._step_count += 1
        if self._step_count % self._steps_per_epoch == 0:
            self.levels = update_levels(
                self.levels,
                self.class_rows,
                self._predict,
                self.gamma,
                self.check_size,
                self.max_strength,
                generator,
            )
            logger.info(
                "cuda levels at step %d: %s", self._step_count, self.levels.tolist()
            )
        return cuda_batch(
            raw, self.levels[y_batch], p_aug, self.max_strength, generator
        )


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
    cuda_params: tuple[float, int, int, int] | None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build the index-batch -> loss closure the training loop calls every step."""
    cuda_state = None
    if method == "cuda":
        assert cuda_params is not None
        cuda_state = _CudaState(model, device, cuda_params)

    def step(idx: torch.Tensor) -> torch.Tensor:
        """Forward one sampled index batch through the encoder and head; return its loss."""
        idx = idx.to(device)
        raw = model.images[idx]
        y_batch = model.y_t[idx]
        if cuda_state is not None:
            raw = cuda_state.augment_batch(raw, y_batch, param, generator)
        x = normalize_batch(raw, model.mean, model.std)
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            features = pool_tokens(model.encoder(x))
            return method_step(features, y_batch)

    return step


def build_loss_step(
    method: str,
    param: float,
    model: StepModel,
    device: torch.device,
    extras: tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        float | None,
        tuple[float, int, int, int] | None,
    ],
    generator: torch.Generator,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build the full index-batch -> loss closure for one stage-one method."""
    log_prior, delta, gcl_s, cuda_params = extras
    method_step = _method_step_fn(
        method, param, model.head, log_prior, delta, gcl_s, generator
    )
    return _build_step_fn(
        method, param, model, device, method_step, generator, cuda_params
    )
