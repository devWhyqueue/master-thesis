from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from imbalance_benchmark.modeling.evaluation import checkpoint_step, initial_checkpoint
from imbalance_benchmark.modeling.models import OkoClassifier
from imbalance_benchmark.modeling.training import (
    CHECKPOINT_INTERVAL,
    resolve_batch_size,
    update_budget,
)

__all__ = ["build_class_index", "sample_oko_sets", "oko_set_loss", "fit_oko"]


def build_class_index(labels: np.ndarray) -> dict[int, list[int]]:
    """Map each class label to the dataset indices that carry it."""
    index: dict[int, list[int]] = {}
    for idx, label in enumerate(labels.tolist()):
        index.setdefault(int(label), []).append(idx)
    return index


def _sample_odd_classes(
    pair_classes: np.ndarray, n_classes: int, rng: np.random.Generator
) -> np.ndarray:
    """Sample one odd class per set uniformly from [C] \\ {pair_class}."""
    raw = rng.integers(n_classes - 1, size=len(pair_classes))
    return np.where(raw < pair_classes, raw, raw + 1)


def _fill_column(
    class_index: dict[int, list[int]],
    classes: np.ndarray,
    set_indices: np.ndarray,
    columns: list[int],
    rng: np.random.Generator,
) -> None:
    """Fill the given columns of set_indices with samples drawn per assigned class."""
    for c, pool in class_index.items():
        mask = classes == c
        count = int(mask.sum())
        if count == 0:
            continue
        pool_arr = np.array(pool)
        for column in columns:
            drawn = rng.integers(len(pool_arr), size=count)
            set_indices[mask, column] = pool_arr[drawn]


def _fill_distinct_pair_indices(
    class_index: dict[int, list[int]],
    pair_classes: np.ndarray,
    set_indices: np.ndarray,
    rng: np.random.Generator,
) -> None:
    """Fill the two same-class set positions without reusing an example."""
    for class_id, pool in class_index.items():
        rows = np.flatnonzero(pair_classes == class_id)
        if not len(rows):
            continue
        if len(pool) < 2:
            raise ValueError("OKO requires two distinct examples in every pair class")
        for row in rows:
            set_indices[row, :2] = rng.choice(pool, size=2, replace=False)


def sample_oko_sets(
    class_index: dict[int, list[int]],
    n_classes: int,
    n_sets: int,
    k: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample n_sets odd-k-out sets (Algorithm 1, Muttenthaler et al. 2024).

    Returns (pair_classes, set_indices of shape (n_sets, k+2), first_odd_classes);
    the auxiliary loss uses only the first odd slot when k > 1.
    """
    pair_classes = rng.integers(n_classes, size=n_sets)
    set_indices = np.empty((n_sets, k + 2), dtype=np.int64)
    _fill_distinct_pair_indices(class_index, pair_classes, set_indices, rng)
    first_odd = _sample_odd_classes(pair_classes, n_classes, rng)
    _fill_column(class_index, first_odd, set_indices, [2], rng)
    for slot in range(1, k):
        odd_col = _sample_odd_classes(pair_classes, n_classes, rng)
        _fill_column(class_index, odd_col, set_indices, [2 + slot], rng)
    return pair_classes, set_indices, first_odd


def oko_set_loss(
    model: OkoClassifier,
    features: torch.Tensor,
    batch_n: int,
    set_size: int,
    pair_labels: torch.Tensor,
    odd_labels: torch.Tensor,
) -> torch.Tensor:
    """OKO hard loss: pair-class CE from the main head plus odd-class CE from the auxiliary head."""
    summed = model.encode(features).view(batch_n, set_size, -1).sum(dim=1)
    return F.cross_entropy(model.main_head(summed), pair_labels) + F.cross_entropy(
        model.odd_head(summed), odd_labels
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
) -> torch.Tensor:
    """Sample one batch of odd-k-out sets and compute their joint OKO loss."""
    pair_classes, set_indices, odd_classes = sample_oko_sets(
        class_index, n_classes, b_size, k, rng
    )
    flat_idx = set_indices.reshape(-1)
    features = torch.stack([dataset[int(i)]["features"] for i in flat_idx]).to(device)
    pair_t = torch.from_numpy(pair_classes).long().to(device)
    odd_t = torch.from_numpy(odd_classes).long().to(device)
    return oko_set_loss(model, features, b_size, k + 2, pair_t, odd_t)


def _oko_train_loop(
    model: OkoClassifier,
    dataset: Any,
    class_index: dict[int, list[int]],
    n_classes: int,
    k: int,
    b_size: int,
    opt: torch.optim.Optimizer,
    ctx: dict[str, Any],
    budget: int,
    rng: np.random.Generator,
    best: dict[str, Any],
) -> dict[str, Any]:
    """Run OKO's update-budgeted training loop, checkpointing on the tie-break rule."""
    device = ctx["device"]
    for step in range(1, budget + 1):
        loss = _oko_step_loss(
            model, dataset, class_index, n_classes, k, b_size, rng, device
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % CHECKPOINT_INTERVAL == 0 or step == budget:
            best = checkpoint_step(
                model, ctx["val_loader"], device, False, n_classes, best
            )
    return best


def fit_oko(ctx: dict[str, Any]) -> tuple[dict[str, Any], float]:
    """OKO: sample odd-k-out sets each step and train the main/auxiliary heads jointly."""
    device = ctx["device"]
    model, dataset, n_classes = ctx["model"], ctx["train_dataset"], ctx["n_classes"]
    k, lr = int(ctx["param_config"]["parameter"]), ctx["param_config"]["lr"]
    b_size = resolve_batch_size(ctx["config"], False)
    budget = update_budget(len(dataset), b_size)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    class_index = build_class_index(ctx["train_labels"])
    rng = np.random.default_rng(ctx["seed"])
    best = initial_checkpoint(model, ctx["val_loader"], device, False, n_classes)
    model.train()
    best = _oko_train_loop(
        model, dataset, class_index, n_classes, k, b_size, opt, ctx, budget, rng, best
    )
    model.load_state_dict({k2: v.to(device) for k2, v in best["state"].items()})
    return best["state"], best["acc"]
