from __future__ import annotations

import math
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
    _RecordingSampler,
)
from imbalance_benchmark.modeling.models import AttentionMil, DualExpertMil, MLP
from imbalance_benchmark.modeling.oko import fit_oko
from imbalance_benchmark.modeling.training import (
    CHECKPOINT_INTERVAL,
    fit_model,
    get_balanced_sampler,
    resolve_batch_size,
    update_budget,
)

__all__ = [
    "fit_crt",
    "fit_rankmix",
    "mde_bag_loss",
    "fit_mde",
    "fit_oko",
    "fit_method",
    "select_post_hoc_tau",
]


def _freeze_and_reinit_classifier(model: nn.Module, is_mil: bool) -> None:
    """Freeze the learned representation and reinitialize only the classifier head (cRT)."""
    if is_mil:
        attn = cast(AttentionMil, model)
        for p in attn.instance_encoder.parameters():
            p.requires_grad_(False)
        for p in attn.attention.parameters():
            p.requires_grad_(False)
        classifier = attn.classifier
    else:
        mlp = cast(MLP, model)
        for p in mlp.net[0].parameters():
            p.requires_grad_(False)
        classifier = cast(nn.Linear, mlp.net[-1])
    nn.init.xavier_uniform_(classifier.weight)
    nn.init.zeros_(classifier.bias)


def fit_crt(ctx: dict[str, Any]) -> tuple[dict[str, Any], float]:
    """cRT: fit CE's inherited config for U updates, then retrain only the reinitialized classifier for ceil(0.2U) updates on a class-balanced loader."""
    device, is_mil = ctx["device"], ctx["is_mil"]
    b_size = resolve_batch_size(ctx["config"], is_mil)
    budget = update_budget(len(ctx["train_dataset"]), b_size)

    stage_one_model = ctx["model_factory"]()
    stage_one_ctx = {
        **ctx,
        "model": stage_one_model,
        "method": "ce",
        "param_config": ctx["stage_one_config"],
    }
    stage_one_state, _ = fit_model(stage_one_ctx, max_steps=budget)
    stage_one_model.load_state_dict(
        {k: v.to(device) for k, v in stage_one_state.items()}
    )
    _freeze_and_reinit_classifier(stage_one_model, is_mil)

    stage_two_ctx = {
        **ctx,
        "model": stage_one_model,
        "method": "balanced_sampling",
        "param_config": {"lr": ctx["param_config"]["lr"], "parameter": 1.0},
    }
    return fit_model(stage_two_ctx, max_steps=math.ceil(0.2 * budget))


def fit_rankmix(ctx: dict[str, Any]) -> tuple[dict[str, Any], float]:
    """RankMix-inspired: train a frozen MIL-CE teacher for U updates, then a reinitialized student on teacher-ranked, mixed bags for U updates."""
    device = ctx["device"]
    b_size = resolve_batch_size(ctx["config"], True)
    budget = update_budget(len(ctx["train_dataset"]), b_size)

    teacher = ctx["model_factory"]()
    teacher_ctx = {
        **ctx,
        "model": teacher,
        "method": "ce",
        "param_config": {"lr": ctx["param_config"]["lr"]},
    }
    teacher_state, _ = fit_model(teacher_ctx, max_steps=budget)
    teacher.load_state_dict({k: v.to(device) for k, v in teacher_state.items()})
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    student = ctx["model_factory"]()
    student_ctx = {**ctx, "model": student, "method": "rankmix", "teacher": teacher}
    return fit_model(student_ctx, max_steps=budget)


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
        batch_size=b_size,
        sampler=_RecordingSampler(natural, exposed),
        collate_fn=bag_collate,
    )
    loader_b = DataLoader(
        dataset,
        batch_size=b_size,
        sampler=_RecordingSampler(
            get_balanced_sampler(train_labels, 1.0, seed + 1), exposed
        ),
        collate_fn=bag_collate,
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
                [b.to(device) for b in bags_u],
                targets_u.to(device),
                [b.to(device) for b in bags_b],
                targets_b.to(device),
                lambda_con,
            )
            loss.backward()
            opt.step()
            step += 1
            if step % CHECKPOINT_INTERVAL == 0 or step == budget:
                best = checkpoint_step(
                    model, ctx["val_loader"], device, True, ctx["n_classes"], best
                )
    return best


def fit_mde(ctx: dict[str, Any]) -> tuple[dict[str, Any], float]:
    """MDE-inspired: U joint updates, each consuming one natural and one class-balanced minibatch."""
    device = ctx["device"]
    model = cast(DualExpertMil, ctx["model"])
    b_size = resolve_batch_size(ctx["config"], True)
    budget = update_budget(len(ctx["train_dataset"]), b_size)
    lr = ctx["param_config"]["lr"]
    lambda_con = float(ctx["param_config"].get("parameter", 0.0))
    loader_u, loader_b = _build_mde_loaders(
        ctx["train_dataset"],
        ctx["train_labels"],
        b_size,
        ctx["seed"],
        ctx.setdefault("exposed_indices", set()),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best = initial_checkpoint(model, ctx["val_loader"], device, True, ctx["n_classes"])
    model.train()
    best = _mde_train_loop(
        model, loader_u, loader_b, opt, ctx, budget, lambda_con, best
    )
    model.load_state_dict({k: v.to(device) for k, v in best["state"].items()})
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
