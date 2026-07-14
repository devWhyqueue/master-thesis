from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch

from imbalance_benchmark.datasets.data import TrainDataset
from imbalance_benchmark.modeling.models import build_model

__all__ = [
    "INPUT_DIM",
    "PATCH_HIDDEN_DIM",
    "MIL_HIDDEN_DIM",
    "DROPOUT",
    "CONDITIONS",
    "LEARNING_RATE_GRID",
    "GRIDS",
    "PATCH_ONLY_METHODS",
    "WSI_ONLY_METHODS",
    "SHARED_METHODS",
    "Regime",
    "roster_for_regime",
    "get_grid_configs",
    "model_kwargs",
    "build_training_ctx",
    "set_training_mode",
    "param_counts",
    "cost_payload",
    "updates_for",
]

INPUT_DIM = 2560
PATCH_HIDDEN_DIM = 512
MIL_HIDDEN_DIM = 256
DROPOUT = 0.1
CONDITIONS = ("natural", "balanced", "moderate", "severe")

LEARNING_RATE_GRID: list[float] = [1e-4, 3e-4, 1e-3, 3e-3]

GRIDS: dict[str, list[float] | list[int]] = {
    "weighted_ce": [0.25, 0.5, 0.75, 1.0],
    "balanced_sampling": [0.25, 0.5, 0.75, 1.0],
    "focal": [0.5, 1.0, 1.5, 2.0],
    "logit_adjustment": [0.25, 0.5, 1.0, 2.0],
    "post_hoc_logit_adjustment": [0.25, 0.5, 1.0, 2.0],
    "ce_soft_f1": [0.25, 1.0, 4.0, 16.0],
    "ce_soft_mcc": [0.25, 1.0, 4.0, 16.0],
    "cfal": [0.25, 1.0, 2.0, 4.0],
    "oko": [1, 2, 4, 8],
    "rankmix": [0.5, 1.0, 2.0, 4.0],
    "sc_mil": [0.05, 0.1, 0.2, 0.5],
    "mde": [0.0, 0.1, 0.25, 0.5],
}

# No imbalance-specific control (Appendix, Table "Experimental Controls"): only the
# common learning-rate grid applies. cRT's stage-two learning rate reuses the same
# four values but is not crossed with anything since stage one inherits CE.
NO_STRENGTH_GRID_METHODS = frozenset({"ce", "crt"})

PATCH_ONLY_METHODS = ("ce_soft_f1", "ce_soft_mcc", "cfal", "oko")
WSI_ONLY_METHODS = ("rankmix", "sc_mil", "mde")
SHARED_METHODS = (
    "ce",
    "balanced_sampling",
    "weighted_ce",
    "focal",
    "logit_adjustment",
    "post_hoc_logit_adjustment",
    "crt",
)


def roster_for_regime(is_mil: bool) -> tuple[str, ...]:
    """Return the prespecified method roster for the WSI or patch regime."""
    return SHARED_METHODS + (WSI_ONLY_METHODS if is_mil else PATCH_ONLY_METHODS)


def get_grid_configs(method: str, n_classes: int | None = None) -> list[dict[str, Any]]:
    """Return the method's candidate configurations per the frozen Appendix grids.

    CE and cRT sweep only the four-value learning-rate grid. Post-hoc logit
    adjustment performs no gradient optimization, so it sweeps only its four
    tau values (no learning-rate crossing) and inherits CE's trained model.
    Every other trainable method crosses its four-value method-specific grid
    with the four-value learning-rate grid (<=16 configurations). OKO's
    odd-class count is additionally capped by K-1 when n_classes is known.
    """
    if method in NO_STRENGTH_GRID_METHODS:
        return [{"lr": lr} for lr in LEARNING_RATE_GRID]
    if method == "post_hoc_logit_adjustment":
        return [{"parameter": p} for p in GRIDS[method]]
    if method not in GRIDS:
        return [{"lr": lr} for lr in LEARNING_RATE_GRID]
    values = GRIDS[method]
    if method == "oko" and n_classes is not None:
        values = sorted({int(v) for v in values if int(v) <= n_classes - 1})
    return [{"parameter": p, "lr": lr} for p in values for lr in LEARNING_RATE_GRID]


def model_kwargs(is_mil: bool) -> dict[str, Any]:
    """Fixed regime-locked architecture arguments shared by every method."""
    return {
        "input_dim": INPUT_DIM,
        "hidden_dim": MIL_HIDDEN_DIM if is_mil else PATCH_HIDDEN_DIM,
        "dropout": DROPOUT,
    }


@dataclass
class Regime:
    """The device, config, and prediction regime shared by one dataset-regime run."""

    device: torch.device
    config: dict[str, Any]
    n_classes: int
    is_mil: bool
    locked_class_names: list[str] = field(default_factory=list, kw_only=True)
    bag_dataset_kwargs: dict[str, int] = field(default_factory=dict, kw_only=True)


def build_training_ctx(
    method: str,
    train_ds: TrainDataset,
    regime: Regime,
    seed: int,
    cfg: dict[str, Any],
    val_loader: torch.utils.data.DataLoader | None = None,
) -> dict[str, Any]:
    """Build the shared training context for one method/config/seed trial."""
    torch.manual_seed(seed)
    kwargs = model_kwargs(regime.is_mil)
    param = cfg.get("parameter")
    _factory = lambda: build_model(
        method, regime.is_mil, n_classes=regime.n_classes, param=param, **kwargs
    ).to(regime.device)
    return {
        "method": method,
        "model": _factory(),
        "model_factory": _factory,
        "train_dataset": train_ds,
        "val_loader": val_loader,
        "device": regime.device,
        "config": regime.config,
        "param_config": cfg,
        "seed": seed,
        "is_mil": regime.is_mil,
        "n_classes": regime.n_classes,
        "train_labels": train_ds.get_int_targets(),
        "exposed_indices": set(),
        "method_diagnostics": {},
        "processed_examples": 0,
    }


def set_training_mode(ctx: dict[str, Any]) -> None:
    """Enable training while preserving deterministic frozen cRT modules."""
    ctx["model"].train()
    for module in ctx.get("frozen_eval_modules", ()):
        module.eval()


def param_counts(model: torch.nn.Module) -> dict[str, int]:
    """Total and trainable parameter counts for one confirmation run's cost record."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_parameters": int(total), "trainable_parameters": int(trainable)}


def cost_payload(
    method: str,
    budget: int,
    elapsed: float,
    model: torch.nn.Module,
    unique_examples: int,
    exposed_examples: int,
    processed_examples: int,
) -> dict[str, Any]:
    """Build an exact confirmation cost record from the examples actually consumed."""
    updates = updates_for(method, budget)
    peak = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
    return {
        "updates": updates,
        "processed_examples": processed_examples,
        "wall_clock_seconds": elapsed,
        "accelerator_hours": elapsed / 3600 if torch.cuda.is_available() else 0.0,
        "peak_accelerator_memory_bytes": peak,
        "examples_per_update": processed_examples / updates if updates else 0.0,
        "unique_training_examples": unique_examples,
        "unique_examples_exposed": exposed_examples,
        "effective_passes_through_unique_examples": processed_examples
        / max(unique_examples, 1),
        **param_counts(model),
    }


def updates_for(method: str, budget: int) -> int:
    """Report's update accounting: RankMix doubles, cRT adds its stage-two budget."""
    if method == "rankmix":
        return 2 * budget
    if method == "crt":
        return budget + math.ceil(0.2 * budget)
    if method == "post_hoc_logit_adjustment":
        return 0
    return budget
