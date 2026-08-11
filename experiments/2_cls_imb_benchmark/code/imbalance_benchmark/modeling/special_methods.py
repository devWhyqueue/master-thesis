from __future__ import annotations

from typing import Any, cast
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from imbalance_benchmark.datasets.data import bag_collate
from imbalance_benchmark.modeling.evaluation import (
    checkpoint_step,
    initial_checkpoint,
    run_evaluation,
    _RecordingBatchSampler,
)
from imbalance_benchmark.modeling.models import DualExpertMil
from imbalance_benchmark.modeling.context import resolve_update_budget
from imbalance_benchmark.modeling.oko import fit_oko
from imbalance_benchmark.modeling.training import (
    build_optimizer,
    fit_model,
    get_balanced_sampler,
    pin_memory_ok,
    resolve_batch_size,
    resolve_checkpoint_interval,
)
from imbalance_benchmark.modeling.workflows.multistage import fit_crt, fit_rankmix

__all__ = [
    "fit_crt",
    "fit_rankmix",
    "mde_bag_loss",
    "fit_mde",
    "fit_oko",
    "fit_method",
    "select_post_hoc_tau",
]


def mde_bag_loss(
    model: DualExpertMil,
    bags_u: list[torch.Tensor],
    targets_u: torch.Tensor,
    bags_b: list[torch.Tensor],
    targets_b: torch.Tensor,
    lambda_con: float,
) -> torch.Tensor:
    """MDE-inspired dual-expert loss: per-branch CE plus cross-expert consistency."""
    emb_u, emb_b = model.aggregate(bags_u), model.aggregate(bags_b)
    logits_u, logits_b = model.logits_u(emb_u), model.logits_b(emb_b)
    loss_cls = F.cross_entropy(logits_u, targets_u) + F.cross_entropy(
        logits_b, targets_b
    )
    if lambda_con <= 0:
        return loss_cls
    logits_u_cross, logits_b_cross = model.logits_b(emb_u), model.logits_u(emb_b)
    loss_con = F.mse_loss(logits_u, logits_u_cross) + F.mse_loss(
        logits_b, logits_b_cross
    )
    return loss_cls + lambda_con * loss_con


def _build_mde_loaders(
    dataset: Any,
    train_labels: np.ndarray,
    b_size: int,
    seed: int,
    exposed: set[int] | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Build MDE's natural and class-balanced bag loaders over the same training pool."""
    exposed = set() if exposed is None else exposed
    natural = torch.utils.data.RandomSampler(
        dataset, generator=torch.Generator().manual_seed(seed)
    )
    loader_u = DataLoader(
        dataset,
        batch_sampler=_RecordingBatchSampler(
            torch.utils.data.BatchSampler(natural, b_size, drop_last=False), exposed
        ),
        collate_fn=bag_collate,
        pin_memory=pin_memory_ok(True),
    )
    loader_b = DataLoader(
        dataset,
        batch_sampler=_RecordingBatchSampler(
            torch.utils.data.BatchSampler(
                get_balanced_sampler(train_labels, 1.0, seed + 1),
                b_size,
                drop_last=False,
            ),
            exposed,
        ),
        collate_fn=bag_collate,
        pin_memory=pin_memory_ok(True),
    )
    return loader_u, loader_b


def _mde_train_loop(
    model: DualExpertMil,
    loader_u: DataLoader,
    loader_b: DataLoader,
    opt: torch.optim.Optimizer,
    ctx: dict[str, Any],
    budget: int,
    lambda_con: float,
    best: dict[str, Any],
) -> dict[str, Any]:
    """Run MDE's U joint updates, each consuming one natural and one balanced minibatch."""
    device = ctx["device"]
    checkpoint_interval = resolve_checkpoint_interval(ctx["config"], True)
    step = 0
    while step < budget:
        for (bags_u, targets_u), (bags_b, targets_b) in zip(
            loader_u, loader_b, strict=False
        ):
            if step >= budget:
                break
            opt.zero_grad()
            loss = mde_bag_loss(
                model,
                [b.to(device, non_blocking=True) for b in bags_u],
                targets_u.to(device, non_blocking=True),
                [b.to(device, non_blocking=True) for b in bags_b],
                targets_b.to(device, non_blocking=True),
                lambda_con,
            )
            ctx["processed_examples"] = (
                ctx.get("processed_examples", 0) + len(targets_u) + len(targets_b)
            )
            ctx["processed_instances"] = ctx.get("processed_instances", 0) + sum(
                len(bag) for bag in [*bags_u, *bags_b]
            )
            loss.backward()
            opt.step()
            step += 1
            if step % checkpoint_interval == 0 or step == budget:
                best = checkpoint_step(
                    model, ctx["val_loader"], device, True, ctx["n_classes"], best, step
                )
    return best


def fit_mde(ctx: dict[str, Any]) -> tuple[dict[str, Any], float]:
    """MDE-inspired: U joint updates, each consuming one natural and one class-balanced minibatch."""
    device = ctx["device"]
    model = cast(DualExpertMil, ctx["model"])
    b_size = resolve_batch_size(ctx["config"], True)
    budget = resolve_update_budget(ctx, b_size)
    lr = ctx["param_config"]["lr"]
    lambda_con = float(ctx["param_config"].get("parameter", 0.0))
    loader_u, loader_b = _build_mde_loaders(
        ctx["train_dataset"],
        ctx["train_labels"],
        b_size,
        ctx["seed"],
        ctx.setdefault("exposed_indices", set()),
    )
    opt = build_optimizer(model.parameters(), lr)
    best = initial_checkpoint(model, ctx["val_loader"], device, True, ctx["n_classes"])
    model.train()
    best = _mde_train_loop(
        model, loader_u, loader_b, opt, ctx, budget, lambda_con, best
    )
    model.load_state_dict({k: v.to(device) for k, v in best["state"].items()})
    ctx["selected_checkpoint_step"] = best["step"]
    return best["state"], best["acc"]


_SPECIAL_FIT = {"crt": fit_crt, "rankmix": fit_rankmix, "mde": fit_mde, "oko": fit_oko}


def fit_method(ctx: dict[str, Any]) -> tuple[dict[str, Any], float]:
    """Dispatch to the method's training orchestration (multi-stage methods route specially)."""
    special = _SPECIAL_FIT.get(ctx["method"])
    return special(ctx) if special is not None else fit_model(ctx)


def select_post_hoc_tau(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    is_mil: bool,
    n_classes: int,
    priors: torch.Tensor,
    taus: list[float],
) -> tuple[float, dict[str, Any]]:
    """Select post-hoc logit-adjustment tau by validation balanced accuracy (no retraining)."""
    best_tau, best_metrics, best_key = taus[0], None, None
    for tau in taus:
        m = run_evaluation(model, val_loader, device, is_mil, n_classes, tau, priors)
        key = (m["balanced_accuracy"], m["macro_f1"], -m["nll"])
        if best_key is None or key > best_key:
            best_key, best_tau, best_metrics = key, tau, m
    assert best_metrics is not None
    return best_tau, best_metrics
