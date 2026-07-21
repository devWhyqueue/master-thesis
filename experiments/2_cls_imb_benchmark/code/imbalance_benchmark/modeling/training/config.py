from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch.utils.data import DataLoader

from imbalance_benchmark.datasets.data import TrainDataset, bag_collate
from imbalance_benchmark.modeling.context import REFERENCE_PASSES, model_kwargs

__all__ = [
    "CHECKPOINT_INTERVAL",
    "OPTIMIZER_NAME",
    "WEIGHT_DECAY",
    "resolve_batch_size",
    "build_optimizer",
    "build_evaluation_loader",
    "resolve_training_config",
]

CHECKPOINT_INTERVAL = 50
# Fixed optimizer family shared by every trainable method and regime.
OPTIMIZER_NAME = "AdamW"
WEIGHT_DECAY = 1e-4
PATCH_EVALUATION_BATCH_SIZE = 4096
MIL_EVALUATION_BATCH_SIZE = 64


def build_evaluation_loader(dataset: TrainDataset, is_mil: bool) -> DataLoader:
    """Build an ordered CPU loader with a regime-appropriate inference batch."""
    return DataLoader(
        dataset,
        batch_size=(
            MIL_EVALUATION_BATCH_SIZE if is_mil else PATCH_EVALUATION_BATCH_SIZE
        ),
        collate_fn=bag_collate if is_mil else None,
        pin_memory=torch.cuda.is_available(),
    )


def resolve_batch_size(cfg: dict[str, Any], is_mil: bool) -> int:
    """Resolve the regime's locked batch size from config."""
    k = "wsi_training" if is_mil else "patch_training"
    sk = "bag_batch_size" if is_mil else "batch_size"
    return cfg.get(k, {}).get(sk, 32 if is_mil else 128)


def build_optimizer(
    params: Iterable[torch.nn.Parameter], lr: float
) -> torch.optim.Optimizer:
    """Construct the single, regime-locked optimizer (AdamW, fixed weight decay).

    Sole construction site so the optimizer family and weight decay recorded by
    :func:`resolve_training_config` cannot drift from what actually trains.
    """
    return torch.optim.AdamW(params, lr=lr, weight_decay=WEIGHT_DECAY)


def resolve_training_config(cfg: dict[str, Any], is_mil: bool) -> dict[str, Any]:
    """Resolved fixed model/optimizer configuration a run actually uses.

    Single source of truth for the values the trainer applies but that never
    appear in the supplied YAML (optimizer family, weight decay, checkpoint
    interval, resolved batch size, budget rule, precision) plus the
    regime-locked architecture. The per-run learning rate and method-specific
    parameter live in the run record's tuning parameters, not here.
    """
    return {
        "optimizer": OPTIMIZER_NAME,
        "weight_decay": WEIGHT_DECAY,
        "checkpoint_interval": CHECKPOINT_INTERVAL,
        "batch_size": resolve_batch_size(cfg, is_mil),
        "update_budget_reference_passes": REFERENCE_PASSES,
        "precision": "float32",
        **model_kwargs(is_mil),
    }
