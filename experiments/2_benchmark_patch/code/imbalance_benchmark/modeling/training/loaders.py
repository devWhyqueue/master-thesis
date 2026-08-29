from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    RandomSampler,
    WeightedRandomSampler,
)

from imbalance_benchmark.datasets.data import bag_collate, patch_collate
from imbalance_benchmark.modeling.evaluation import (
    ClassAwareBatchSampler,
    _RecordingBatchSampler,
)
from imbalance_benchmark.modeling.training.config import pin_memory_ok

__all__ = [
    "FIXED_BALANCED_SAMPLER_METHODS",
    "get_balanced_sampler",
    "build_train_loader",
]

# Scholz sampling-loss hybrids: class-balanced oversampling plus a metric loss.
FIXED_BALANCED_SAMPLER_METHODS = frozenset({"ce_soft_f1", "ce_soft_mcc"})


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


def _loader(ctx: dict[str, Any], sampler: Any, is_mil: bool) -> DataLoader:
    return DataLoader(
        ctx["train_dataset"],
        batch_sampler=sampler,
        collate_fn=bag_collate if is_mil else patch_collate,  # type: ignore[arg-type]
        pin_memory=pin_memory_ok(is_mil),
    )


def _base_sampler(
    ctx: dict[str, Any], train_labels: np.ndarray, param: float | None
) -> RandomSampler | WeightedRandomSampler:
    method = ctx["method"]
    # Falsy check intentional: strength 0 must give RandomSampler (CE's anchor),
    # not a with-replacement uniform sampler. Do not change to `is not None`.
    if method == "balanced_sampling" and param:
        return get_balanced_sampler(train_labels, param, ctx["seed"])
    if method in FIXED_BALANCED_SAMPLER_METHODS:
        return get_balanced_sampler(train_labels, 1.0, ctx["seed"])
    return RandomSampler(
        ctx["train_dataset"], generator=torch.Generator().manual_seed(ctx["seed"])
    )


def build_train_loader(
    ctx: dict[str, Any],
    train_labels: np.ndarray,
    param: float | None,
    b_size: int,
    is_mil: bool,
) -> DataLoader:
    """Build the condition's train loader: class-aware for SC-MIL, sampled otherwise."""
    exposed = (
        ctx.setdefault("exposed_indices", set())
        if ctx.get("record_exposure", True)
        else None
    )
    if ctx["method"] == "sc_mil":
        sampler = _RecordingBatchSampler(
            ClassAwareBatchSampler(train_labels, b_size, ctx["seed"]), exposed
        )
        return _loader(ctx, sampler, is_mil)
    base = _base_sampler(ctx, train_labels, param)
    sampler = _RecordingBatchSampler(
        BatchSampler(base, b_size, drop_last=False), exposed
    )
    return _loader(ctx, sampler, is_mil)
