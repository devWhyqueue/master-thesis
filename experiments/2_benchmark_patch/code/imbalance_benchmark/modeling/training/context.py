from __future__ import annotations

from typing import Any

import torch

from imbalance_benchmark.datasets.data import TrainDataset
from imbalance_benchmark.modeling.context import Regime, model_kwargs
from imbalance_benchmark.modeling.models import build_model


def build_training_ctx(
    method: str,
    train_ds: TrainDataset,
    regime: Regime,
    seed: int,
    cfg: dict[str, Any],
    val_loader: torch.utils.data.DataLoader | None = None,
    example_budget: int | None = None,
) -> dict[str, Any]:
    """Build shared training context for one method/config/seed trial."""
    torch.manual_seed(seed)
    kwargs, param = model_kwargs(regime.is_mil), cfg.get("parameter")

    def factory() -> torch.nn.Module:
        """Construct a fresh model with this run's locked settings."""
        return build_model(
            method, regime.is_mil, n_classes=regime.n_classes, param=param, **kwargs
        ).to(regime.device)

    context = _training_context(
        method, train_ds, regime, seed, cfg, val_loader, factory
    )
    if example_budget is not None:
        context["example_budget"] = int(example_budget)
    return context


def _training_context(
    method: str,
    train_ds: TrainDataset,
    regime: Regime,
    seed: int,
    cfg: dict[str, Any],
    val_loader: torch.utils.data.DataLoader | None,
    factory: Any,
) -> dict[str, Any]:
    """Build method-independent training state."""
    return {
        "method": method,
        "model": factory(),
        "model_factory": factory,
        "train_dataset": train_ds,
        "val_loader": val_loader,
        "device": regime.device,
        "config": regime.config,
        "param_config": cfg,
        "seed": seed,
        "is_mil": regime.is_mil,
        "n_classes": regime.n_classes,
        "train_labels": train_ds.get_int_targets(),
        "difficulty": regime.difficulty,
        "exposed_indices": set(),
        "method_diagnostics": {},
        "processed_examples": 0,
    }
