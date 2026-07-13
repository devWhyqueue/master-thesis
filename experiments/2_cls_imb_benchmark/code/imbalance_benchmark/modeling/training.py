from __future__ import annotations

import logging
import math
from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from imbalance_benchmark.datasets.data import bag_collate
from imbalance_benchmark.modeling.evaluation import (
    checkpoint_step,
    initial_checkpoint,
    run_evaluation,
)
from imbalance_benchmark.modeling.losses import (
    FocalLoss,
    ScholzCombinedLoss,
    cfal_loss,
    rankmix_bag_loss,
    supervised_contrastive_loss,
)

logger = logging.getLogger(__name__)

__all__ = [
    "CHECKPOINT_INTERVAL",
    "FIXED_BALANCED_SAMPLER_METHODS",
    "get_class_weights",
    "get_balanced_sampler",
    "class_priors",
    "run_evaluation",
    "update_budget",
    "resolve_batch_size",
    "fit_model",
]

FIXED_BALANCED_SAMPLER_METHODS = frozenset({"sc_mil", "rankmix"})
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
    return torch.tensor(
        np.bincount(labels, minlength=n_classes) / len(labels), dtype=torch.float32
    ).to(device)


def update_budget(support: int, batch_size: int) -> int:
    """U = 30 * ceil(T / B): 30 reference passes through the controlled support."""
    return 30 * math.ceil(support / batch_size)


def resolve_batch_size(cfg: dict[str, Any], is_mil: bool) -> int:
    """Resolve the regime's locked batch size from config."""
    key, size_key = (
        ("wsi_training", "bag_batch_size")
        if is_mil
        else ("patch_training", "batch_size")
    )
    return cfg.get(key, {}).get(size_key, 32 if is_mil else 128)


def _fit_step(
    batch_data: Any, ctx: dict[str, Any], step: int, max_steps: int
) -> torch.Tensor:
    """Execute a single forward-backward update step for single-loader methods."""
    is_mil, method, device = ctx["is_mil"], ctx["method"], ctx["device"]
    model, criterion, param = ctx["model"], ctx["criterion"], ctx["param"]
    if is_mil:
        bags, targets = batch_data
        bags, targets = [b.to(device) for b in bags], targets.to(device)
        if method == "rankmix":
            loss, _ = rankmix_bag_loss(model, ctx["teacher"], bags, targets, param)
            return loss
        if method == "sc_mil":
            logits, emb, _ = model.forward_bags(bags)
            cont, _ = supervised_contrastive_loss(
                model.project_bag_embeddings(emb), targets, temperature=param
            )
            return (1.0 - (step / max_steps)) * cont + (
                step / max_steps
            ) * F.cross_entropy(logits, targets)
        logits, _, _ = model.forward_bags(bags)
        if method == "logit_adjustment" and param is not None:
            return F.cross_entropy(
                logits + param * torch.log(ctx["priors"] + 1e-8), targets
            )
        return criterion(logits, targets)
    features, targets = (
        batch_data["features"].to(device),
        batch_data["target"].to(device),
    )
    if method == "cfal":
        return cfal_loss(model, features, targets, ctx["class_counts"])
    if method == "logit_adjustment" and param is not None:
        return F.cross_entropy(
            model(features) + param * torch.log(ctx["priors"] + 1e-8), targets
        )
    return criterion(model(features), targets)


def _init_criterion(
    method: str,
    param: float | None,
    n_classes: int,
    train_labels: np.ndarray,
    device: torch.device,
) -> nn.Module:
    """Initialize the loss criterion for training."""
    if method == "weighted_ce" and param is not None:
        return nn.CrossEntropyLoss(
            weight=get_class_weights(train_labels, n_classes, strength=param).to(device)
        )
    if method == "focal" and param is not None:
        return FocalLoss(gamma=param)
    if method in ("ce_soft_f1", "ce_soft_mcc") and param is not None:
        return ScholzCombinedLoss(
            n_classes, metric="f1" if "f1" in method else "mcc", weight=param
        )
    return nn.CrossEntropyLoss()


def _build_train_loader(
    ctx: dict[str, Any],
    train_labels: np.ndarray,
    param: float | None,
    b_size: int,
    is_mil: bool,
) -> DataLoader:
    """Build the training loader, applying the balanced sampler when required."""
    method = ctx["method"]
    if method == "balanced_sampling" and param:
        sampler = get_balanced_sampler(train_labels, param, ctx["seed"])
    elif method in FIXED_BALANCED_SAMPLER_METHODS:
        sampler = get_balanced_sampler(train_labels, 1.0, ctx["seed"])
    else:
        sampler = None
    return DataLoader(
        ctx["train_dataset"],
        batch_size=b_size,
        sampler=sampler,
        shuffle=sampler is None,
        collate_fn=bag_collate if is_mil else None,
        generator=torch.Generator().manual_seed(ctx["seed"]),
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
    step = 0
    device, is_mil, n_classes = ctx["device"], ctx["is_mil"], ctx["n_classes"]
    while step < max_steps:
        for batch in train_loader:
            if step >= max_steps:
                break
            optimizer.zero_grad()
            _fit_step(batch, ctx, step, max_steps).backward()
            optimizer.step()
            step += 1
            if step % CHECKPOINT_INTERVAL == 0 or step == max_steps:
                best = checkpoint_step(
                    ctx["model"], val_loader, device, is_mil, n_classes, best
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
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    _prepare_training_context(ctx, param, device)

    budget = (
        max_steps
        if max_steps is not None
        else update_budget(len(ctx["train_dataset"]), b_size)
    )
    best = _run_training_loop(opt, loader, ctx["val_loader"], ctx, budget, best)
    model.load_state_dict({k: v.to(device) for k, v in best["state"].items()})
    return best["state"], best["acc"]
