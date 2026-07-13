from __future__ import annotations

import logging
import math
from typing import Any, cast
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from imbalance_benchmark.data import bag_collate
from imbalance_benchmark.modeling.evaluation import run_evaluation
from imbalance_benchmark.modeling.models import DualExpertMil
from imbalance_benchmark.modeling.losses import (
    FocalLoss,
    ScholzCombinedLoss,
    cfal_loss,
    supervised_contrastive_loss,
)

logger = logging.getLogger(__name__)

__all__ = ["get_class_weights", "get_balanced_sampler", "run_evaluation", "fit_model"]


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
        [float(w[l] ** strength) for l in labels],
        len(labels),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )


def _fit_step(
    batch_data: Any, ctx: dict[str, Any], step: int, max_steps: int
) -> torch.Tensor:
    """Execute a single forward-backward update step."""
    is_mil, method, device = ctx["is_mil"], ctx["method"], ctx["device"]
    model, criterion, param = ctx["model"], ctx["criterion"], ctx["param"]
    if is_mil:
        bags, targets = batch_data
        bags, targets = [b.to(device) for b in bags], targets.to(device)
        if method == "mde":
            expert = cast(DualExpertMil, model)
            emb = expert.aggregate(bags)
            loss = F.cross_entropy(expert.logits_u(emb), targets)
            return (
                loss + param * F.mse_loss(expert.logits_u(emb), expert.logits_b(emb))
                if (param is not None and param > 0)
                else loss
            )
        elif method == "sc_mil":
            logits, emb, _ = model.forward_bags(bags)
            cont, _ = supervised_contrastive_loss(
                model.project_bag_embeddings(emb), targets, temperature=0.1
            )
            return (1.0 - (step / max_steps)) * cont + (
                step / max_steps
            ) * F.cross_entropy(logits, targets)
        return F.cross_entropy(model.forward_bags(bags)[0], targets)
    features, targets = (
        batch_data["features"].to(device),
        batch_data["target"].to(device),
    )
    if method == "cfal" and param is not None:
        return cfal_loss(model, features, targets, ctx["labels"], margin=param)
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


def _run_training_loop(
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    ctx: dict[str, Any],
    max_steps: int,
    best_state: dict[str, Any],
) -> tuple[dict[str, Any], float]:
    """Execute training step loop and validate checkpoints."""
    step, best_acc, best_f1, best_nll = 0, -1.0, -1.0, float("inf")
    device, is_mil, priors = ctx["device"], ctx["is_mil"], ctx["priors"]
    while step < max_steps:
        for batch in train_loader:
            if step >= max_steps:
                break
            optimizer.zero_grad()
            _fit_step(batch, ctx, step, max_steps).backward()
            optimizer.step()
            step += 1
            if val_loader is not None and (step % 50 == 0 or step == max_steps):
                m = run_evaluation(
                    ctx["model"], val_loader, device, is_mil, len(priors)
                )
                acc, f1, nll = m["balanced_accuracy"], m["macro_f1"], m["nll"]
                if acc > best_acc or (
                    abs(acc - best_acc) < 1e-6
                    and (f1 > best_f1 or (abs(f1 - best_f1) < 1e-6 and nll < best_nll))
                ):
                    best_acc, best_f1, best_nll = acc, f1, nll
                    best_state = {
                        k: v.cpu().clone() for k, v in ctx["model"].state_dict().items()
                    }
    return best_state, best_acc


def _resolve_batch_size(cfg: dict[str, Any], is_mil: bool) -> int:
    """Resolve the regime's locked batch size from config."""
    key, size_key = (
        ("wsi_training", "bag_batch_size")
        if is_mil
        else ("patch_training", "batch_size")
    )
    return cfg.get(key, {}).get(size_key, 32 if is_mil else 128)


def _build_train_loader(
    ctx: dict[str, Any],
    train_labels: np.ndarray,
    param: float | None,
    b_size: int,
    is_mil: bool,
) -> DataLoader:
    """Build the training loader, applying the balanced sampler when requested."""
    sampler = (
        get_balanced_sampler(train_labels, param, ctx["seed"])
        if ctx["method"] == "balanced_sampling" and param
        else None
    )
    return DataLoader(
        ctx["train_dataset"],
        batch_size=b_size,
        sampler=sampler,
        shuffle=sampler is None,
        collate_fn=bag_collate if is_mil else None,
    )


def _initial_checkpoint(
    model: nn.Module,
    val_loader: DataLoader | None,
    device: torch.device,
    is_mil: bool,
    n_classes: int,
) -> dict[str, Any]:
    """Snapshot the untrained model as the initial best-checkpoint state."""
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    if val_loader is not None:
        run_evaluation(model, val_loader, device, is_mil, n_classes)
    return best_state


def _prepare_training_context(
    ctx: dict[str, Any], param: float | None, device: torch.device
) -> None:
    """Attach the per-run criterion, class priors, and tuned parameter to the context."""
    n_classes, train_labels = ctx["n_classes"], ctx["train_labels"]
    ctx["param"] = param
    ctx["priors"] = torch.tensor(
        np.bincount(train_labels, minlength=n_classes) / len(train_labels),
        dtype=torch.float32,
    ).to(device)
    ctx["criterion"] = _init_criterion(
        ctx["method"], param, n_classes, train_labels, device
    )


def fit_model(ctx: dict[str, Any]) -> tuple[dict[str, Any], float]:
    """Train the model for a fixed updates budget and return the best checkpoint state."""
    model, device, is_mil = ctx["model"], ctx["device"], ctx["is_mil"]
    train_labels, param_config = ctx["train_labels"], ctx["param_config"]
    lr, param = param_config["lr"], param_config.get("parameter")
    b_size = _resolve_batch_size(ctx["config"], is_mil)
    loader = _build_train_loader(ctx, train_labels, param, b_size, is_mil)

    best_state = _initial_checkpoint(
        model, ctx["val_loader"], device, is_mil, ctx["n_classes"]
    )
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    _prepare_training_context(ctx, param, device)

    max_steps = int(30 * math.ceil(len(ctx["train_dataset"]) / b_size))
    best_state, final_acc = _run_training_loop(
        opt, loader, ctx["val_loader"], ctx, max_steps, best_state
    )
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return best_state, final_acc
