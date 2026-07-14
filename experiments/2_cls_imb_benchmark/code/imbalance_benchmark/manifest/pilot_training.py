from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from imbalance_benchmark.datasets.data import BagFeatureDataset, ImbalanceDataset
from imbalance_benchmark.modeling.models import AttentionMil, MLP
from imbalance_benchmark.modeling.training import fit_model


def fit_pilot_model(
    scratch_path: Path,
    device: torch.device,
    n_classes: int,
    is_mil: bool,
    val_loader: torch.utils.data.DataLoader,
    initialization_seed: int,
) -> tuple[nn.Module, float]:
    """Construct reproducible pilot weights and fit the fixed CE baseline."""
    dataset_class = BagFeatureDataset if is_mil else ImbalanceDataset
    dataset = dataset_class(scratch_path, device=device)
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
        "config": {},
        "param_config": {"lr": 1e-3},
        "seed": initialization_seed,
        "is_mil": is_mil,
        "n_classes": n_classes,
        "train_labels": dataset.get_int_targets(),
    }
    _, best_accuracy = fit_model(context)
    return model, best_accuracy
