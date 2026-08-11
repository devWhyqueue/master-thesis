from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from imbalance_benchmark.modeling.evaluation import checkpoint_step, initial_checkpoint
from imbalance_benchmark.modeling.context import resolve_update_budget
from imbalance_benchmark.modeling.models import OkoClassifier
from imbalance_benchmark.modeling.oko.sampling import (
    OkoPools,
    _build_oko_pools,
    _independent_units,
    build_class_index,
    sample_oko_sets,
)
from imbalance_benchmark.modeling.training import (
    build_optimizer,
    resolve_batch_size,
    resolve_checkpoint_interval,
)

__all__ = [
    "build_class_index",
    "sample_oko_sets",
    "oko_set_loss",
    "fit_oko",
]


def oko_set_loss(
    model: OkoClassifier,
    features: torch.Tensor,
    batch_n: int,
    set_size: int,
    pair_labels: torch.Tensor,
    odd_labels: torch.Tensor,
) -> torch.Tensor:
    """OKO hard loss over aggregated set logits (main + auxiliary odd head).

    The report defines the set logits as the sum of per-example head outputs,
    ``f_theta(S) = sum_i f_theta(x_i)``; summing the hidden embeddings first and
    applying the biased head once would undercount the head bias.
    """
    encoded = model.encode(features).view(batch_n, set_size, -1)
    main_logits = model.main_head(encoded).sum(dim=1)
    odd_logits = model.odd_head(encoded).sum(dim=1)
    return F.cross_entropy(main_logits, pair_labels) + F.cross_entropy(
        odd_logits, odd_labels
    )


def _oko_step_loss(
    model: OkoClassifier,
    dataset: Any,
    class_index: dict[int, list[int]],
    n_classes: int,
    k: int,
    b_size: int,
    rng: np.random.Generator,
    device: torch.device,
    pools: OkoPools,
    exposed: set[int] | None = None,
) -> torch.Tensor:
    """Sample one batch of odd-k-out sets and compute their joint OKO loss."""
    pair_classes, set_indices, odd_classes = sample_oko_sets(
        class_index, n_classes, b_size, k, rng, pools=pools
    )
    flat_idx = set_indices.reshape(-1)
    if exposed is not None:
        exposed.update(int(i) for i in flat_idx)
    features = dataset.__getitems__(flat_idx.tolist())["features"].to(device)
    pair_t = torch.from_numpy(pair_classes).long().to(device)
    odd_t = torch.from_numpy(odd_classes).long().to(device)
    return oko_set_loss(model, features, b_size, k + 2, pair_t, odd_t)


def _oko_train_loop(
    opt: torch.optim.Optimizer,
    ctx: dict[str, Any],
    budget: int,
    best: dict[str, Any],
    class_index: dict[int, list[int]],
    pools: OkoPools,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Run OKO's update-budgeted training loop, checkpointing on the tie-break rule."""
    model, dataset, n_classes = ctx["model"], ctx["train_dataset"], ctx["n_classes"]
    device, exposed = ctx["device"], ctx.setdefault("exposed_indices", set())
    k = int(ctx["param_config"]["parameter"])
    b_size = resolve_batch_size(ctx["config"], False)
    checkpoint_interval = resolve_checkpoint_interval(ctx["config"], False)
    for step in range(1, budget + 1):
        loss = _oko_step_loss(
            model,
            dataset,
            class_index,
            n_classes,
            k,
            b_size,
            rng,
            device,
            pools,
            exposed,
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % checkpoint_interval == 0 or step == budget:
            best = checkpoint_step(
                model, ctx["val_loader"], device, False, n_classes, best, step
            )
            logging.info("tune: oko seed=%s step %d/%d", ctx.get("seed"), step, budget)
    return best


def fit_oko(ctx: dict[str, Any]) -> tuple[dict[str, Any], float]:
    """OKO: sample odd-k-out sets each step and train the main/auxiliary heads jointly."""
    device = ctx["device"]
    model, dataset, n_classes = ctx["model"], ctx["train_dataset"], ctx["n_classes"]
    k, lr = int(ctx["param_config"]["parameter"]), ctx["param_config"]["lr"]
    b_size = resolve_batch_size(ctx["config"], False)
    budget = resolve_update_budget(ctx, b_size)
    opt = build_optimizer(model.parameters(), lr)
    class_index = build_class_index(ctx["train_labels"])
    pools = _build_oko_pools(class_index, _independent_units(dataset))
    rng = np.random.default_rng(ctx["seed"])
    best = initial_checkpoint(model, ctx["val_loader"], device, False, n_classes)
    model.train()
    best = _oko_train_loop(opt, ctx, budget, best, class_index, pools, rng)
    model.load_state_dict({k2: v.to(device) for k2, v in best["state"].items()})
    ctx["processed_examples"] = budget * b_size * (k + 2)
    ctx["selected_checkpoint_step"] = best["step"]
    return best["state"], best["acc"]
