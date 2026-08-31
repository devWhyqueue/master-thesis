from __future__ import annotations

from collections.abc import Iterable
import math
from typing import Any

import torch
from torch.utils.data import DataLoader

from imbalance_benchmark.datasets.data import TrainDataset, bag_collate, patch_collate
from imbalance_benchmark.datasets.features.cache import bank_is_cpu
from imbalance_benchmark.modeling.context import REFERENCE_PASSES, model_kwargs

__all__ = [
    "TARGET_CHECKPOINTS",
    "OPTIMIZER_NAME",
    "WEIGHT_DECAY",
    "resolve_batch_size",
    "resolve_checkpoint_schedule",
    "build_optimizer",
    "build_evaluation_loader",
    "pin_memory_ok",
    "resolve_training_config",
]

TARGET_CHECKPOINTS = 170
# Fixed optimizer family shared by every trainable method and regime.
OPTIMIZER_NAME = "AdamW"
WEIGHT_DECAY = 1e-4
PATCH_EVALUATION_BATCH_SIZE = 131072
MIL_EVALUATION_BATCH_SIZE = 64


def pin_memory_ok(is_mil: bool) -> bool:
    """Whether a loader may pin its output batch.

    Pinning a tensor already resident on CUDA raises; MIL never reads the
    device-resident feature bank, so only the patch-regime bank placement
    matters here.
    """
    return torch.cuda.is_available() and (is_mil or bank_is_cpu())


def build_evaluation_loader(dataset: TrainDataset, is_mil: bool) -> DataLoader:
    """Build an ordered CPU loader with a regime-appropriate inference batch."""
    return DataLoader(
        dataset,
        batch_size=(
            MIL_EVALUATION_BATCH_SIZE if is_mil else PATCH_EVALUATION_BATCH_SIZE
        ),
        collate_fn=bag_collate if is_mil else patch_collate,  # type: ignore[arg-type]
        pin_memory=pin_memory_ok(is_mil),
    )


def resolve_batch_size(cfg: dict[str, Any], is_mil: bool) -> int:
    """Resolve the regime's locked batch size from config."""
    k = "wsi_training" if is_mil else "patch_training"
    sk = "bag_batch_size" if is_mil else "batch_size"
    return cfg.get(k, {}).get(sk, 32 if is_mil else 128)


def _log_spaced_positions(count: int) -> list[float]:
    """Positions in (0, 1], front-loaded: dense near 0, sparse near 1."""
    if count <= 1:
        return [1.0]
    denom = math.log1p(count - 1)
    return [1 - math.log1p(count - 1 - i) / denom for i in range(count)]


def resolve_checkpoint_schedule(budget: int) -> frozenset[int]:
    """Validation steps for one fit: log-spaced, ~TARGET_CHECKPOINTS of them.

    Same target count as the prior uniform cadence, but front-loaded rather
    than evenly spaced -- confirmation's own selected-checkpoint distribution
    (2940 runs) put 83% of winners in the first quarter of budget and only 1%
    in the final tenth, so a uniform grid wasted most of its resolution where
    it was least useful. Below ``TARGET_CHECKPOINTS`` steps, every step is a
    checkpoint (nothing to compress); a fit's own final step is always one.
    """
    if budget <= TARGET_CHECKPOINTS:
        return frozenset(range(1, budget + 1))
    positions = _log_spaced_positions(TARGET_CHECKPOINTS)
    return frozenset(max(1, min(budget, round(p * budget))) for p in positions)


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
    cadence rule, resolved batch size, budget rule, precision) plus the
    regime-locked architecture. The per-run learning rate and method-specific
    parameter live in the run record's tuning parameters, not here.

    ``target_checkpoints`` records the cadence rule rather than a single
    resolved interval: each condition derives its own interval from its own
    update budget, so no single resolved number is representative here.
    """
    return {
        "optimizer": OPTIMIZER_NAME,
        "weight_decay": WEIGHT_DECAY,
        "target_checkpoints": TARGET_CHECKPOINTS,
        "batch_size": resolve_batch_size(cfg, is_mil),
        "budget_unit": "example_presentations",
        "example_budget_reference_passes": REFERENCE_PASSES,
        "precision": "float32",
        **model_kwargs(is_mil),
    }
