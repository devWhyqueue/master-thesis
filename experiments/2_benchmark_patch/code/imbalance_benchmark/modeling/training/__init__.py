from __future__ import annotations

import logging
from typing import Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from imbalance_benchmark.modeling.context import (
    GRIDS,
    NO_STRENGTH_GRID_METHODS,
    REFERENCE_PASSES,
    set_training_mode,
)
from imbalance_benchmark.modeling.training.budget import (
    example_budget as _example_budget,
    resolve_update_budget,
    updates_for_exposure,
)
from imbalance_benchmark.modeling.training.config import (
    build_evaluation_loader,
    build_optimizer,
    pin_memory_ok,
    resolve_batch_size,
    resolve_checkpoint_schedule,
)
from imbalance_benchmark.modeling.evaluation import (
    checkpoint_step,
    evaluate_metrics,
    initial_checkpoint,
    run_evaluation,
    ClassAwareBatchSampler,
)
from imbalance_benchmark.modeling.losses import (
    FocalLoss,
    ScholzCombinedLoss,
    cfal_loss,
)
from imbalance_benchmark.modeling.training.loaders import (
    FIXED_BALANCED_SAMPLER_METHODS,
    build_train_loader,
    get_balanced_sampler,
)
from imbalance_benchmark.modeling.training.mil import _fit_mil_step
from imbalance_benchmark.modeling.training.semantic_scale import (
    prepare_ssb_pool,
    ssb_loss,
)
from imbalance_benchmark.modeling.training.signal_weights import (
    mean_one as _mean_one,
    signal_criterion,
)

logger = logging.getLogger(__name__)

__all__ = [
    "FIXED_BALANCED_SAMPLER_METHODS",
    "get_class_weights",
    "get_balanced_sampler",
    "ClassAwareBatchSampler",
    "class_priors",
    "run_evaluation",
    "example_budget",
    "updates_for_exposure",
    "resolve_batch_size",
    "resolve_checkpoint_schedule",
    "build_optimizer",
    "build_evaluation_loader",
    "pin_memory_ok",
    "fit_model",
]


def get_class_weights(
    labels: np.ndarray, n_classes: int, strength: float = 1.0
) -> torch.Tensor:
    """Compute rescaled inverse-frequency weights."""
    counts = np.maximum(np.bincount(labels, minlength=n_classes), 1.0)
    return _mean_one(np.power(1.0 / counts, strength))


def class_priors(
    labels: np.ndarray, n_classes: int, device: torch.device
) -> torch.Tensor:
    """Compute the realized per-class training prior."""
    p = np.bincount(labels, minlength=n_classes) / len(labels)
    return torch.tensor(p, dtype=torch.float32).to(device)


def example_budget(support: int) -> int:
    """E = REFERENCE_PASSES * T example presentations."""
    return _example_budget(support, REFERENCE_PASSES)


def _init_criterion(
    method: str,
    param: float | None,
    n_classes: int,
    train_labels: np.ndarray,
    device: torch.device,
    ctx: dict[str, Any],
) -> nn.Module:
    """Instantiate the loss function according to the method config."""
    if method in GRIDS and method not in NO_STRENGTH_GRID_METHODS and param is None:
        raise ValueError(
            f"{method} has a strength dimension but its locked parameter is None"
        )
    if method == "weighted_ce":
        strength = float(param if param is not None else 1.0)
        w = get_class_weights(train_labels, n_classes, strength)
        return nn.CrossEntropyLoss(weight=w.to(device))
    if method == "focal":
        alpha = get_class_weights(train_labels, n_classes, 1.0)
        return FocalLoss(gamma=float(param if param is not None else 1.0), alpha=alpha)
    if method in ("ce_soft_f1", "ce_soft_mcc"):
        metric = "f1" if "f1" in method else "mcc"
        return ScholzCombinedLoss(
            n_classes, metric=metric, weight=float(param if param is not None else 1.0)
        )
    signal = signal_criterion(method, param, device, ctx)
    return signal if signal is not None else nn.CrossEntropyLoss()


def _fit_step(
    batch_data: Any, ctx: dict[str, Any], step: int, max_steps: int
) -> torch.Tensor:
    """Execute a single forward-backward update step for single-loader methods."""
    method, device = ctx["method"], ctx["device"]
    model, criterion, param = ctx["model"], ctx["criterion"], ctx["param"]
    if ctx["is_mil"]:
        return _fit_mil_step(batch_data, ctx, step, max_steps)
    inputs = batch_data["features"].to(device, non_blocking=True)
    targets = batch_data["target"].to(device, non_blocking=True)
    ctx["processed_examples"] = ctx.get("processed_examples", 0) + len(targets)
    if method == "cfal":
        return cfal_loss(model, inputs, targets, ctx["class_counts"])
    if method == "semantic_scale_ce":
        return ssb_loss(model, inputs, targets, ctx, step)
    logits = model(inputs)
    if method == "logit_adjustment" and param is not None:
        logits = logits + param * torch.log(ctx["priors"] + 1e-8)
    if method == "mde":
        return criterion(model.encode(inputs), logits, targets)
    return criterion(logits, targets)


def _run_training_loop(
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    ctx: dict[str, Any],
    max_steps: int,
    best: dict[str, Any],
) -> dict[str, Any]:
    """Execute the update-budgeted training loop, checkpointing on the tie-break rule.

    ``ctx["dense_trace"]``, when set to a list, additionally records every
    ``ctx["dense_trace_interval"]``-th step's raw validation metrics -- used
    only by the offline checkpoint-cadence/truncation gate (plan
    after-a-first-run-linear-wave item 4), never by production tuning or
    confirmation, which leave it unset and see unchanged behavior.
    """
    step, device, is_mil, n_classes = 0, ctx["device"], ctx["is_mil"], ctx["n_classes"]
    checkpoint_schedule = resolve_checkpoint_schedule(max_steps)
    trace = ctx.get("dense_trace")
    trace_interval = ctx.get("dense_trace_interval", 1)
    while step < max_steps:
        for batch in train_loader:
            if step >= max_steps:
                break
            optimizer.zero_grad()
            set_training_mode(ctx)
            _fit_step(batch, ctx, step, max_steps).backward()
            optimizer.step()
            step += 1
            if trace is not None and (step % trace_interval == 0 or step == max_steps):
                metrics = evaluate_metrics(
                    ctx["model"], val_loader, device, is_mil, n_classes
                )
                if metrics is not None:
                    trace.append({"step": step, **metrics})
            if step in checkpoint_schedule:
                best = checkpoint_step(
                    ctx["model"], val_loader, device, is_mil, n_classes, best, step
                )
                logger.info(
                    "tune: %s seed=%s step %d/%d",
                    ctx["method"],
                    ctx.get("seed"),
                    step,
                    max_steps,
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
        ctx["method"], param, n_classes, train_labels, device, ctx
    )


def fit_model(
    ctx: dict[str, Any], max_steps: int | None = None
) -> tuple[dict[str, Any], float]:
    """Train a single-loader method for its update budget; return the best checkpoint."""
    model, device, is_mil = ctx["model"], ctx["device"], ctx["is_mil"]
    train_labels, param_config = ctx["train_labels"], ctx["param_config"]
    lr, param = param_config["lr"], param_config.get("parameter")
    b_size = resolve_batch_size(ctx["config"], is_mil)
    loader = build_train_loader(ctx, train_labels, param, b_size, is_mil)
    best = initial_checkpoint(
        model, ctx["val_loader"], device, is_mil, ctx["n_classes"]
    )
    set_training_mode(ctx)
    opt = build_optimizer(model.parameters(), lr)
    _prepare_training_context(ctx, param, device)
    prepare_ssb_pool(ctx, b_size)
    budget = (
        max_steps
        if max_steps is not None
        else resolve_update_budget(
            ctx, ctx["method"], param_config, b_size, REFERENCE_PASSES
        )
    )
    best = _run_training_loop(opt, loader, ctx["val_loader"], ctx, budget, best)
    model.load_state_dict({k: v.to(device) for k, v in best["state"].items()})
    ctx["selected_checkpoint_step"] = best["step"]
    return best["state"], best["acc"]
