"""Stage-one training: LoRA-adapted Virchow2 + linear/cosine head, one method per run.

``ce``, ``bs``, ``cuda``, and ``la`` share a plain linear head; ``cuda`` augments the
raw uint8 patch batch before the encoder forward, at each class's current level of
learning; ``gcl`` uses a cosine head and the clouded-logit loss. Balanced sampling
(``bs``) is the only stage-one method that changes the row sampler; every other
method draws uniformly.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
import torch
import torch.nn as nn

from mitigation import TrainHParams, cuda_constants, gcl_constants, train_hparams
from mitigation.encoder import load_lora_encoder
from mitigation.images import decode_images, decode_transform
from mitigation.methods import CosineHead, gcl_delta, sample_weights
from mitigation.train.loop import run_training_steps
from mitigation.train.stage1_step import StepModel, build_loss_step

__all__ = ["ArmBatch", "Stage1Output", "run_stage1"]


class ArmBatch(NamedTuple):
    """One arm's training rows: raw patch paths, integer targets, and per-class counts."""

    image_paths: list[str]
    y: np.ndarray
    counts: np.ndarray
    num_classes: int


@dataclass
class Stage1Output:
    """A trained stage-one encoder and head, ready for evaluation and export."""

    encoder: nn.Module
    head: nn.Module
    data_config: dict[str, Any]
    cosine_scale: float | None
    images: torch.Tensor


def _row_weights(
    y: np.ndarray, num_classes: int, method: str, param: float
) -> np.ndarray:
    """Balanced sampling reweights rows by ``param``; every other method draws uniformly."""
    return sample_weights(y, num_classes, "power", param if method == "bs" else 0.0)


def _build_optimizer(
    encoder: nn.Module, head: nn.Module, hparams: TrainHParams
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    """AdamW over the trainable LoRA + head parameters, with a cosine LR schedule."""
    trainable = [p for p in encoder.parameters() if p.requires_grad] + list(
        head.parameters()
    )
    optimizer = torch.optim.AdamW(
        trainable,
        lr=hparams.lr,
        weight_decay=hparams.weight_decay,
        fused=trainable[0].device.type == "cuda",
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=hparams.steps
    )
    return optimizer, scheduler


def _method_extras(
    method: str, counts: np.ndarray, config: dict[str, Any], device: torch.device
) -> tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    float | None,
    tuple[float, int, int, int] | None,
]:
    """Build the method-specific tensors ``run_stage1`` needs.

    LA's log-prior, GCL's delta/scale, CUDA's (gamma, check_size, max_strength,
    steps_per_epoch) -- the epoch length in optimizer steps, since training runs a
    fixed step count rather than epochs (report "CUDA" thesis variant).
    """
    if method == "la":
        prior = torch.tensor(counts / counts.sum(), dtype=torch.float32, device=device)
        return torch.log(prior), None, None, None
    if method == "gcl":
        gcl_s, _, _ = gcl_constants(config)
        delta = torch.tensor(gcl_delta(counts), dtype=torch.float32, device=device)
        return None, delta, gcl_s, None
    if method == "cuda":
        gamma, check_size, max_strength = cuda_constants(config)
        batch_size = train_hparams(config, "stage1").batch_size
        steps_per_epoch = -(-int(counts.sum()) // batch_size)
        return None, None, None, (gamma, check_size, max_strength, steps_per_epoch)
    return None, None, None, None


def _build_head(
    pooled_dim: int, num_classes: int, is_cosine: bool, device: torch.device
) -> nn.Module:
    """A linear head, or a cosine head for GCL, sized to the encoder's pooled feature width."""
    head = (
        CosineHead(pooled_dim, num_classes)
        if is_cosine
        else nn.Linear(pooled_dim, num_classes)
    )
    return head.to(device)


@dataclass
class _TrainingSetup:
    """Everything ``run_stage1`` builds before it can start the training loop."""

    encoder: nn.Module
    head: nn.Module
    data_config: dict[str, Any]
    images: torch.Tensor
    y_t: torch.Tensor
    cpu_generator: torch.Generator
    device_generator: torch.Generator
    is_cosine: bool


def _prepare(
    config: dict[str, Any], device: torch.device, arm: ArmBatch, method: str, seed: int
) -> tuple[
    _TrainingSetup,
    torch.Tensor | None,
    torch.Tensor | None,
    float | None,
    tuple[float, int, int, int] | None,
]:
    """Load the LoRA encoder, decode the arm's images, and build its head and loss extras."""
    cpu_generator = torch.Generator().manual_seed(seed)
    device_generator = torch.Generator(device=device).manual_seed(seed)
    encoder, data_config, pooled_dim = load_lora_encoder(config, device)
    images = decode_images(arm.image_paths, decode_transform(data_config)).to(device)
    y_t = torch.from_numpy(arm.y).to(device)
    is_cosine = method == "gcl"
    head = _build_head(pooled_dim, arm.num_classes, is_cosine, device)
    log_prior, delta, gcl_s, cuda_params = _method_extras(
        method, arm.counts, config, device
    )
    setup = _TrainingSetup(
        encoder,
        head,
        data_config,
        images,
        y_t,
        cpu_generator,
        device_generator,
        is_cosine,
    )
    return setup, log_prior, delta, gcl_s, cuda_params


def _run_training(
    config: dict[str, Any],
    setup: _TrainingSetup,
    arm: ArmBatch,
    method: str,
    param: float,
    step_fn: Callable[[torch.Tensor], torch.Tensor],
) -> None:
    """Build the optimizer/schedule and run the fixed-step training loop."""
    hparams = train_hparams(config, "stage1")
    optimizer, scheduler = _build_optimizer(setup.encoder, setup.head, hparams)
    weights = _row_weights(arm.y, arm.num_classes, method, param)
    run_training_steps(
        weights,
        hparams.steps,
        hparams.batch_size,
        optimizer,
        scheduler,
        setup.cpu_generator,
        step_fn,
    )


def run_stage1(
    config: dict[str, Any],
    device: torch.device,
    arm: ArmBatch,
    method: str,
    param: float,
    seed: int,
) -> Stage1Output:
    """Train one stage-one arm: LoRA-adapted encoder plus a linear or cosine head."""
    setup, log_prior, delta, gcl_s, cuda_params = _prepare(
        config, device, arm, method, seed
    )
    mean, std = tuple(setup.data_config["mean"]), tuple(setup.data_config["std"])
    model = StepModel(setup.encoder, setup.head, setup.images, setup.y_t, mean, std)
    step_fn = build_loss_step(
        method,
        param,
        model,
        device,
        (log_prior, delta, gcl_s, cuda_params),
        setup.device_generator,
    )
    _run_training(config, setup, arm, method, param, step_fn)
    cosine_scale = gcl_s if setup.is_cosine else None
    return Stage1Output(
        setup.encoder, setup.head, setup.data_config, cosine_scale, setup.images
    )
