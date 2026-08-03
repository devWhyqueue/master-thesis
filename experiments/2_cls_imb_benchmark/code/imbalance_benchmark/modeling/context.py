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
    "REFERENCE_PASSES",
    "CONDITIONS",
    "CONTROLLED_CONDITIONS",
    "NATURAL_ANCHOR_METHODS",
    "LEARNING_RATE_GRID",
    "GRIDS",
    "PATCH_ONLY_METHODS",
    "WSI_ONLY_METHODS",
    "SHARED_METHODS",
    "Regime",
    "roster_for_regime",
    "roster_for_condition",
    "group_conditions",
    "get_grid_configs",
    "model_kwargs",
    "build_training_ctx",
    "resolve_update_budget",
    "set_training_mode",
]

INPUT_DIM = 2560
PATCH_HIDDEN_DIM = 512
MIL_HIDDEN_DIM = 256
DROPOUT = 0.1
# Update budget U = REFERENCE_PASSES * ceil(T / B) (report §"Model training and selection").
REFERENCE_PASSES = 30
CONDITIONS = ("natural", "balanced", "moderate", "severe")
CONTROLLED_CONDITIONS = ("balanced", "moderate", "severe")

# The natural anchor is descriptive and never enters the imbalance deficit or
# recovery estimands (report §"From imbalance deficit to mitigation recovery"),
# so only the CE reference is fitted there.
NATURAL_ANCHOR_METHODS = ("ce",)

# Current-centered window into workflows.tuning.search_windows.LR_ENVELOPE[2:6].
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
    "sc_mil": [0.05, 0.1, 0.5, 1.0],
    "mde": [0.0, 0.1, 0.25, 0.5],
}

# No imbalance-specific control (Appendix, Table "Experimental Controls"):
# only the common learning-rate grid applies.
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


def roster_for_condition(is_mil: bool, condition: str) -> tuple[str, ...]:
    """Return the methods fitted in one training condition.

    Mitigation is compared only against a size-matched balanced reference, so
    the full roster applies to the controlled conditions alone; the natural
    anchor fits ``NATURAL_ANCHOR_METHODS``.
    """
    if condition == "natural":
        return NATURAL_ANCHOR_METHODS
    return roster_for_regime(is_mil)


def group_conditions(group: str) -> tuple[str, ...]:
    """Return the conditions one SLURM partition group covers."""
    if group == "natural":
        return ("natural",)
    if group == "controlled":
        return CONTROLLED_CONDITIONS
    raise ValueError(f"Unknown condition group: {group}")


def get_grid_configs(
    method: str,
    n_classes: int | None = None,
    lr_window: list[float] | None = None,
    strength_window: list[float] | None = None,
) -> list[dict[str, Any]]:
    """Return the method's candidates for one active search window (default: frozen).

    CE/cRT sweep only ``lr_window``; post-hoc adjustment sweeps only its taus;
    other methods cross ``strength_window`` with ``lr_window``, capping OKO by K-1.
    """
    lr_values = lr_window if lr_window is not None else LEARNING_RATE_GRID
    if method in NO_STRENGTH_GRID_METHODS:
        return [{"lr": lr} for lr in lr_values]
    if method == "post_hoc_logit_adjustment":
        return [{"parameter": p} for p in GRIDS[method]]
    if method not in GRIDS:
        return [{"lr": lr} for lr in lr_values]
    values = strength_window if strength_window is not None else GRIDS[method]
    if method == "oko" and n_classes is not None:
        values = sorted({int(v) for v in values if int(v) <= n_classes - 1})
    return [{"parameter": p, "lr": lr} for p in values for lr in lr_values]


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
    method_grids: dict[str, list[dict[str, Any]]] = field(
        default_factory=dict, kw_only=True
    )
    update_budgets: dict[str, int] = field(default_factory=dict, kw_only=True)


def build_training_ctx(
    method: str,
    train_ds: TrainDataset,
    regime: Regime,
    seed: int,
    cfg: dict[str, Any],
    val_loader: torch.utils.data.DataLoader | None = None,
    update_budget: int | None = None,
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
        **({"update_budget": int(update_budget)} if update_budget is not None else {}),
    }


def resolve_update_budget(ctx: dict[str, Any], batch_size: int) -> int:
    """Use the signed update budget when present, otherwise retain pilot fallback."""
    fallback = REFERENCE_PASSES * math.ceil(len(ctx["train_dataset"]) / batch_size)
    return int(ctx.get("update_budget", fallback))


def set_training_mode(ctx: dict[str, Any]) -> None:
    """Enable training while preserving deterministic frozen cRT modules."""
    ctx["model"].train()
    for module in ctx.get("frozen_eval_modules", ()):
        module.eval()
