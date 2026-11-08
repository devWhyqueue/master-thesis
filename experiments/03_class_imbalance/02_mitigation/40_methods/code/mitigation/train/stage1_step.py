"""Per-step loss closures for stage-one training: one per method, sharing one encoder forward."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, NamedTuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mitigation.encoder import pool_tokens
from mitigation.images import normalize_batch
from mitigation.methods import cuda_batch, gcl_loss, la_loss, update_levels

__all__ = [
    "CudaParams",
    "Stage1Extras",
    "StepModel",
    "build_loss_step",
    "class_weight_tensor",
]

logger = logging.getLogger(__name__)

# (gamma, check_size, max_strength, steps_per_epoch, batch_size)
CudaParams = tuple[float, int, int, int, int]


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
        params: CudaParams,
    ) -> None:
        """``params`` is (gamma, check_size, max_strength, steps_per_epoch, batch_size) (Eq. cuda-lol)."""
        self._model = model
        self._device = device
        self.gamma, self.check_size, self.max_strength = params[:3]
        self._steps_per_epoch, self._batch_size = params[3:]
        num_classes = int(model.y_t.max().item()) + 1
        self.levels = torch.zeros(num_classes, dtype=torch.long, device=device)
        self.class_rows = [model.images[model.y_t == c] for c in range(num_classes)]
        self._rows_seen = 0

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
        """Replace each row by its class's level-of-learning augmentation (Eq. cuda-aug).

        Levels update on the first (micro-)batch of every ``steps_per_epoch``-th optimizer
        step; rows seen, not calls, count the steps, so micro-batching keeps the timing.
        """
        step, offset = divmod(self._rows_seen, self._batch_size)
        self._rows_seen += len(raw)
        if offset == 0 and (step + 1) % self._steps_per_epoch == 0:
            self.levels = update_levels(
                self.levels,
                self.class_rows,
                self._predict,
                self.gamma,
                self.check_size,
                self.max_strength,
                generator,
            )
            logger.info("cuda levels at step %d: %s", step + 1, self.levels.tolist())
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


class Stage1Extras(NamedTuple):
    """Opt-in stage-one options: fixed CE class weights and a training-audit dict to fill."""

    class_weights: np.ndarray | None = None
    audit: dict[str, Any] | None = None


def class_weight_tensor(
    class_weights: np.ndarray | None, num_classes: int, device: torch.device
) -> torch.Tensor | None:
    """Validate optional fixed CE class weights: one finite positive value per class."""
    if class_weights is None:
        return None
    weights = np.asarray(class_weights, dtype=np.float64)
    if weights.shape != (num_classes,):
        raise ValueError(f"need {num_classes} class weights, got shape {weights.shape}")
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError(f"class weights must be finite and positive: {weights}")
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _weighted_ce_step_fn(
    method: str, head: nn.Module, class_weights: torch.Tensor
) -> Any:
    """Fixed-class-weight CE: per-row CE times its class weight, averaged over the batch size.

    Never divides by the batch's weight sum, so a batch missing a class keeps the fixed
    dataset-level normalization (plain CE only).
    """
    if method != "ce":
        raise ValueError(f"class weights apply to plain CE only, not {method!r}")
    return lambda features, y: (
        class_weights[y] * F.cross_entropy(head(features), y, reduction="none")
    ).mean()


def _build_step_fn(
    method: str,
    param: float,
    model: StepModel,
    device: torch.device,
    method_step: Any,
    generator: torch.Generator,
    cuda_params: CudaParams | None,
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
        CudaParams | None,
    ],
    generator: torch.Generator,
    *,
    class_weights: torch.Tensor | None = None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build the full index-batch -> loss closure for one stage-one method."""
    log_prior, delta, gcl_s, cuda_params = extras
    method_step = (
        _method_step_fn(method, param, model.head, log_prior, delta, gcl_s, generator)
        if class_weights is None
        else _weighted_ce_step_fn(method, model.head, class_weights)
    )
    return _build_step_fn(
        method, param, model, device, method_step, generator, cuda_params
    )
