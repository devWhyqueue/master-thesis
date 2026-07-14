from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from imbalance_benchmark.datasets.data import BagFeatureDataset, ImbalanceDataset
from imbalance_benchmark.modeling.models import AttentionMil, MLP
from imbalance_benchmark.modeling.training import fit_model


def _pilot_dataset(
    scratch_path: Path,
    device: torch.device,
    is_mil: bool,
    bag_kwargs: dict[str, int] | None,
) -> BagFeatureDataset | ImbalanceDataset:
    """Load the pilot's training manifest with the regime's fixed evidence controls."""
    if is_mil:
        controls = bag_kwargs or {}
        return BagFeatureDataset(
            scratch_path,
            max_instances=controls.get("max_instances", 500),
            instance_selection_seed=controls.get("instance_selection_seed", 0),
            device=device,
        )
    return ImbalanceDataset(scratch_path, device=device)


def fit_pilot_model(
    scratch_path: Path,
    device: torch.device,
    n_classes: int,
    is_mil: bool,
    val_loader: torch.utils.data.DataLoader,
    initialization_seed: int,
    config: dict[str, Any] | None = None,
    bag_kwargs: dict[str, int] | None = None,
) -> tuple[nn.Module, float]:
    """Construct reproducible pilot weights and fit the fixed CE baseline."""
    dataset = _pilot_dataset(scratch_path, device, is_mil, bag_kwargs)
    torch.manual_seed(initialization_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(initialization_seed)
    hidden_dim = 256 if is_mil else 512
    model = (AttentionMil if is_mil else MLP)(2560, hidden_dim, n_classes, 0.1).to(
        device
    )
    context: dict[str, Any] = {
        "method": "ce",
        "model": model,
        "train_dataset": dataset,
        "val_loader": val_loader,
        "device": device,
        "config": config or {},
        "param_config": {"lr": 1e-3},
        "seed": initialization_seed,
        "is_mil": is_mil,
        "n_classes": n_classes,
        "train_labels": dataset.get_int_targets(),
    }
    _, best_accuracy = fit_model(context)
    return model, best_accuracy
