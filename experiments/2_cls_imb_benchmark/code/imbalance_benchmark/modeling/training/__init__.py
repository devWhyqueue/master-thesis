from __future__ import annotations

import logging
import math
from typing import Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, WeightedRandomSampler

from imbalance_benchmark.datasets.data import bag_collate
from imbalance_benchmark.modeling.context import (
    resolve_update_budget,
    set_training_mode,
)
from imbalance_benchmark.modeling.evaluation import (
    checkpoint_step,
    initial_checkpoint,
    run_evaluation,
    ClassAwareBatchSampler,
    _RecordingSampler,
    _RecordingBatchSampler,
)
from imbalance_benchmark.modeling.losses import (
    FocalLoss,
    ScholzCombinedLoss,
    cfal_loss,
)
from imbalance_benchmark.modeling.training.mil import _fit_mil_step

logger = logging.getLogger(__name__)

__all__ = [
    "CHECKPOINT_INTERVAL",
    "FIXED_BALANCED_SAMPLER_METHODS",
    "get_class_weights",
    "get_balanced_sampler",
    "ClassAwareBatchSampler",
    "class_priors",
    "run_evaluation",
    "update_budget",
    "resolve_batch_size",
    "fit_model",
]

# Scholz sampling-loss hybrids: class-balanced oversampling plus a metric loss.
FIXED_BALANCED_SAMPLER_METHODS = frozenset({"ce_soft_f1", "ce_soft_mcc"})
CHECKPOINT_INTERVAL = 50


def get_class_weights(
    labels: np.ndarray, n_classes: int, strength: float = 1.0
) -> torch.Tensor:
    """Compute rescaled inverse-frequency weights."""
    w = torch.tensor(
        np.power(
            1.0 / np.maximum(np.bincount(labels, minlength=n_classes), 1.0), strength
        ),
        dtype=torch.float32,
    )
    return w * (n_classes / w.sum())


def get_balanced_sampler(
    labels: np.ndarray, strength: float = 1.0, seed: int = 0
) -> WeightedRandomSampler:
    """Create a WeightedRandomSampler for class balancing."""
    w = 1.0 / np.maximum(np.bincount(labels), 1.0)
    return WeightedRandomSampler(
        [float(w[label] ** strength) for label in labels],
        len(labels),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )


def class_priors(
    labels: np.ndarray, n_classes: int, device: torch.device
) -> torch.Tensor:
    """Compute the realized per-class training prior."""
    p = np.bincount(labels, minlength=n_classes) / len(labels)
    return torch.tensor(p, dtype=torch.float32).to(device)


def update_budget(support: int, batch_size: int) -> int:
    """U = 30 * ceil(T / B): 30 reference passes through the controlled support."""
    return 30 * math.ceil(support / batch_size)


def resolve_batch_size(cfg: dict[str, Any], is_mil: bool) -> int:
    """Resolve the regime's locked batch size from config."""
    k = "wsi_training" if is_mil else "patch_training"
    sk = "bag_batch_size" if is_mil else "batch_size"
    return cfg.get(k, {}).get(sk, 32 if is_mil else 128)


def _init_criterion(
    method: str,
    param: float | None,
    n_classes: int,
    train_labels: np.ndarray,
    device: torch.device,
) -> nn.Module:
    """Instantiate the loss function according to the method config."""
    if method == "weighted_ce":
        w = get_class_weights(train_labels, n_classes, float(param or 1.0))
        return nn.CrossEntropyLoss(weight=w.to(device))
    if method == "focal":
        return FocalLoss(gamma=float(param or 1.0))
    if method in ("ce_soft_f1", "ce_soft_mcc"):
        metric = "f1" if "f1" in method else "mcc"
        return ScholzCombinedLoss(n_classes, metric=metric, weight=float(param or 1.0))
    return nn.CrossEntropyLoss()


def _fit_step(
    batch_data: Any, ctx: dict[str, Any], step: int, max_steps: int
) -> torch.Tensor:
    """Execute a single forward-backward update step for single-loader methods."""
    method, device = ctx["method"], ctx["device"]
    model, criterion, param = ctx["model"], ctx["criterion"], ctx["param"]
    if ctx["is_mil"]:
        return _fit_mil_step(batch_data, ctx, step, max_steps)
    inputs, targets = batch_data["features"].to(device), batch_data["target"].to(device)
    ctx["processed_examples"] = ctx.get("processed_examples", 0) + len(targets)
    if method == "cfal":
        return cfal_loss(model, inputs, targets, ctx["class_counts"])
    logits = model(inputs)
    if method == "logit_adjustment" and param is not None:
        logits = logits + param * torch.log(ctx["priors"] + 1e-8)
    if method == "mde":
        return criterion(model.encode(inputs), logits, targets)
    return criterion(logits, targets)


def _build_train_loader(
    ctx: dict[str, Any],
    train_labels: np.ndarray,
    param: float | None,
    b_size: int,
    is_mil: bool,
) -> DataLoader:
    method, exposed = ctx["method"], ctx.setdefault("exposed_indices", set())
    if method == "sc_mil":
        sampler = _RecordingBatchSampler(
            ClassAwareBatchSampler(train_labels, b_size, ctx["seed"]), exposed
        )
        return DataLoader(
            ctx["train_dataset"],
            batch_sampler=sampler,
            collate_fn=bag_collate if is_mil else None,
        )
    gen = torch.Generator().manual_seed(ctx["seed"])
    if method == "balanced_sampling" and param:
        base = get_balanced_sampler(train_labels, param, ctx["seed"])
    elif method in FIXED_BALANCED_SAMPLER_METHODS:
        base = get_balanced_sampler(train_labels, 1.0, ctx["seed"])
    else:
        base = RandomSampler(ctx["train_dataset"], generator=gen)
    return DataLoader(
        ctx["train_dataset"],
        batch_size=b_size,
        sampler=_RecordingSampler(base, exposed),
        collate_fn=bag_collate if is_mil else None,
    )


def _run_training_loop(
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    ctx: dict[str, Any],
    max_steps: int,
    best: dict[str, Any],
) -> dict[str, Any]:
    """Execute the update-budgeted training loop, checkpointing on the tie-break rule."""
    step, device, is_mil, n_classes = 0, ctx["device"], ctx["is_mil"], ctx["n_classes"]
    while step < max_steps:
        for batch in train_loader:
            if step >= max_steps:
                break
            optimizer.zero_grad()
            set_training_mode(ctx)
            _fit_step(batch, ctx, step, max_steps).backward()
            optimizer.step()
            step += 1
            if step % CHECKPOINT_INTERVAL == 0 or step == max_steps:
                best = checkpoint_step(
                    ctx["model"], val_loader, device, is_mil, n_classes, best, step
                )
    return best


def _prepare_training_context(
    ctx: dict[str, Any], param: float | None, device: torch.device
) -> None:
    """Attach the per-run criterion, class priors/counts, and tuned parameter to the context."""
    n_classes, train_labels = ctx["n_classes"], ctx["train_labels"]
    ctx["param"] = param
    ctx["priors"] = class_priors(train_labels, n_classes, device)
    ctx["class_counts"] = np.bincount(train_labels, minlength=n_classes)
    ctx["criterion"] = _init_criterion(
        ctx["method"], param, n_classes, train_labels, device
    )


def fit_model(
    ctx: dict[str, Any], max_steps: int | None = None
) -> tuple[dict[str, Any], float]:
    """Train a single-loader method for its update budget; return the best checkpoint."""
    model, device, is_mil = ctx["model"], ctx["device"], ctx["is_mil"]
    train_labels, param_config = ctx["train_labels"], ctx["param_config"]
    lr, param = param_config["lr"], param_config.get("parameter")
    b_size = resolve_batch_size(ctx["config"], is_mil)
    loader = _build_train_loader(ctx, train_labels, param, b_size, is_mil)
    best = initial_checkpoint(
        model, ctx["val_loader"], device, is_mil, ctx["n_classes"]
    )
    set_training_mode(ctx)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    _prepare_training_context(ctx, param, device)
    budget = max_steps if max_steps is not None else resolve_update_budget(ctx, b_size)
    best = _run_training_loop(opt, loader, ctx["val_loader"], ctx, budget, best)
    model.load_state_dict({k: v.to(device) for k, v in best["state"].items()})
    ctx["selected_checkpoint_step"] = best["step"]
    return best["state"], best["acc"]
