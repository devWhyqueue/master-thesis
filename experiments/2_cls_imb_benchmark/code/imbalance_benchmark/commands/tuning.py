from __future__ import annotations

import argparse
import json
from typing import Any

import torch

from imbalance_benchmark.common import (
    ensure_dirs,
    get_grid_configs,
    load_config,
    write_json,
)
from imbalance_benchmark.data import ImbalanceDataset
from imbalance_benchmark.manifest.freeze import verify_manifest_freeze
from imbalance_benchmark.modeling.models import MLP
from imbalance_benchmark.modeling.training import fit_model

__all__ = ["cmd_tune"]


def _tune_method(
    method: str,
    train_ds: ImbalanceDataset,
    val_ldr: torch.utils.data.DataLoader,
    device: torch.device,
    config: dict[str, Any],
    seed: int,
    n_cls: int,
) -> dict[str, Any]:
    """Tune hyperparameters for a single method."""
    configs = get_grid_configs(method)
    best_acc, best_cfg = -1.0, configs[0]
    for cfg in configs[:2]:
        ctx = {
            "method": method,
            "model": MLP(2560, 256, n_cls, 0.1).to(device),
            "train_dataset": train_ds,
            "val_loader": val_ldr,
            "device": device,
            "config": config,
            "param_config": cfg,
            "seed": seed,
            "is_mil": False,
            "n_classes": n_cls,
            "train_labels": train_ds.get_int_targets(),
        }
        _, acc = fit_model(ctx)
        if acc > best_acc:
            best_acc, best_cfg = acc, cfg
    return best_cfg


def cmd_tune(args: argparse.Namespace) -> None:
    """Run validation tuning sweep."""
    config = load_config(args.config)
    paths = ensure_dirs(config)
    freeze_path = paths["data"] / "manifest_freeze.json"
    if freeze_path.exists():
        verify_manifest_freeze(json.loads(freeze_path.read_text()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_ds = ImbalanceDataset(
        paths["data"] / "manifest.csv", split_name="validation", device=device
    )
    n_cls = val_ds.get_n_classes()
    train_ds = ImbalanceDataset(paths["data"] / "manifest_balanced.csv", device=device)
    best = {
        m: _tune_method(
            m,
            train_ds,
            torch.utils.data.DataLoader(val_ds, batch_size=64),
            device,
            config,
            args.seed,
            n_cls,
        )
        for m in ["ce", "weighted_ce", "focal"]
    }
    write_json(paths["data"] / "tuning_selections.json", best)
