from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from imbalance_benchmark.modeling.evaluation import checkpoint_step, initial_checkpoint
from imbalance_benchmark.modeling.context import resolve_update_budget
from imbalance_benchmark.modeling.models import OkoClassifier
from imbalance_benchmark.modeling.training import (
    CHECKPOINT_INTERVAL,
    build_optimizer,
    resolve_batch_size,
)

__all__ = ["build_class_index", "sample_oko_sets", "oko_set_loss", "fit_oko"]


def build_class_index(labels: np.ndarray) -> dict[int, list[int]]:
    """Map each class label to the dataset indices that carry it."""
    index: dict[int, list[int]] = {}
    for idx, label in enumerate(labels.tolist()):
        index.setdefault(int(label), []).append(idx)
    return index


def _independent_units(dataset: Any) -> np.ndarray | None:
    """Independent-unit id (patient, or slide) per patch, aligned to dataset order."""
    frame = getattr(dataset, "df", None)
    if frame is None or "case_id" not in getattr(frame, "columns", []):
        return None
    return frame["case_id"].to_numpy()


def _sample_distinct_odd_classes(
    pair_classes: np.ndarray, n_classes: int, k: int, rng: np.random.Generator
) -> np.ndarray:
    """Sample k distinct odd classes per set (report requirement) from [C] \\ {pair}."""
    out = np.empty((len(pair_classes), k), dtype=np.int64)
    for row, pair in enumerate(pair_classes.tolist()):
        others = np.array([c for c in range(n_classes) if c != pair])
        out[row] = rng.choice(others, size=k, replace=False)
    return out


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


def _class_unit_pools(
    class_index: dict[int, list[int]], units: np.ndarray
) -> dict[int, dict[Any, list[int]]]:
    """Group each class's example indices by their independent unit (patient/slide)."""
    pools: dict[int, dict[Any, list[int]]] = {}
    for class_id, indices in class_index.items():
        by_unit: dict[Any, list[int]] = {}
        for idx in indices:
            by_unit.setdefault(units[idx], []).append(idx)
        pools[class_id] = by_unit
    return pools


def _draw_distinct_unit_pair(
    by_unit: dict[Any, list[int]], rng: np.random.Generator
) -> tuple[int, int]:
    """Draw two same-class examples from two distinct independent units."""
    unit_keys = list(by_unit)
    first_unit, second_unit = rng.choice(len(unit_keys), size=2, replace=False)
    first = rng.choice(by_unit[unit_keys[first_unit]])
    second = rng.choice(by_unit[unit_keys[second_unit]])
    return int(first), int(second)


def _fill_distinct_pair_indices(
    class_index: dict[int, list[int]],
    pair_classes: np.ndarray,
    set_indices: np.ndarray,
    rng: np.random.Generator,
    units: np.ndarray | None = None,
) -> None:
    """Fill the two same-class positions from two distinct independent units.

    The report requires class-aware batches to hold two distinct same-class
    *independent units* (patients/slides), not merely two distinct example
    indices which could be two patches from the same patient/slide. When no unit
    map is supplied the two examples are only guaranteed distinct.
    """
    unit_pools = _class_unit_pools(class_index, units) if units is not None else None
    for class_id, pool in class_index.items():
        rows = np.flatnonzero(pair_classes == class_id)
        if not len(rows):
            continue
        if unit_pools is not None:
            by_unit = unit_pools[class_id]
            if len(by_unit) < 2:
                raise ValueError(
                    "OKO requires two distinct same-class independent units"
                )
            for row in rows:
                set_indices[row, :2] = _draw_distinct_unit_pair(by_unit, rng)
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
    units: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample n_sets odd-k-out sets (Algorithm 1, Muttenthaler et al. 2024).

    Returns (pair_classes, set_indices of shape (n_sets, k+2), first_odd_classes);
    the auxiliary loss uses only the first odd slot. With ``units``, each set's
    two same-class examples come from distinct independent units.
    """
    pair_classes = rng.integers(n_classes, size=n_sets)
    set_indices = np.empty((n_sets, k + 2), dtype=np.int64)
    _fill_distinct_pair_indices(class_index, pair_classes, set_indices, rng, units)
    odd_classes = _sample_distinct_odd_classes(pair_classes, n_classes, k, rng)
    for slot in range(k):
        _fill_column(class_index, odd_classes[:, slot], set_indices, [2 + slot], rng)
    return pair_classes, set_indices, odd_classes[:, 0]


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
    units: np.ndarray | None = None,
    exposed: set[int] | None = None,
) -> torch.Tensor:
    """Sample one batch of odd-k-out sets and compute their joint OKO loss."""
    pair_classes, set_indices, odd_classes = sample_oko_sets(
        class_index, n_classes, b_size, k, rng, units
    )
    flat_idx = set_indices.reshape(-1)
    if exposed is not None:
        exposed.update(int(i) for i in flat_idx)
    features = torch.stack([dataset[int(i)]["features"] for i in flat_idx]).to(device)
    pair_t = torch.from_numpy(pair_classes).long().to(device)
    odd_t = torch.from_numpy(odd_classes).long().to(device)
    return oko_set_loss(model, features, b_size, k + 2, pair_t, odd_t)


def _oko_train_loop(
    opt: torch.optim.Optimizer,
    ctx: dict[str, Any],
    budget: int,
    best: dict[str, Any],
    class_index: dict[int, list[int]],
    units: np.ndarray | None,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Run OKO's update-budgeted training loop, checkpointing on the tie-break rule."""
    model, dataset, n_classes = ctx["model"], ctx["train_dataset"], ctx["n_classes"]
    device, exposed = ctx["device"], ctx.setdefault("exposed_indices", set())
    k = int(ctx["param_config"]["parameter"])
    b_size = resolve_batch_size(ctx["config"], False)
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
            units,
            exposed,
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % CHECKPOINT_INTERVAL == 0 or step == budget:
            best = checkpoint_step(
                model, ctx["val_loader"], device, False, n_classes, best, step
            )
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
    units = _independent_units(dataset)
    rng = np.random.default_rng(ctx["seed"])
    best = initial_checkpoint(model, ctx["val_loader"], device, False, n_classes)
    model.train()
    best = _oko_train_loop(opt, ctx, budget, best, class_index, units, rng)
    model.load_state_dict({k2: v.to(device) for k2, v in best["state"].items()})
    ctx["processed_examples"] = budget * b_size * (k + 2)
    ctx["selected_checkpoint_step"] = best["step"]
    return best["state"], best["acc"]
