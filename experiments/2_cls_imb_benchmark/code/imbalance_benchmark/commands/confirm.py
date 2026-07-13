from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from imbalance_benchmark.common import ensure_dirs, load_config, write_run_record
from imbalance_benchmark.data import ImbalanceDataset
from imbalance_benchmark.modeling.models import MLP
from imbalance_benchmark.modeling.training import fit_model, run_evaluation

__all__ = ["cmd_confirm"]


def _confirm_run(
    cond: str,
    method: str,
    cfg: dict[str, Any],
    train_ds: ImbalanceDataset,
    test_ldr: torch.utils.data.DataLoader,
    device: torch.device,
    config: dict[str, Any],
    paths: dict[str, Path],
    n_cls: int,
) -> None:
    """Train confirmation models for a single condition-method configuration across 5 seeds."""
    for seed in range(5):
        model = MLP(2560, 256, n_cls, 0.1).to(device)
        ctx = {
            "method": method,
            "model": model,
            "train_dataset": train_ds,
            "val_loader": None,
            "device": device,
            "config": config,
            "param_config": cfg,
            "seed": seed,
            "is_mil": False,
            "n_classes": n_cls,
            "train_labels": train_ds.get_int_targets(),
        }
        fit_model(ctx)
        res = run_evaluation(model, test_ldr, device, False, n_cls)
        write_run_record(
            paths["results"] / cond / method / f"seed={seed}",
            {
                "benchmark": "patch",
                "method": method,
                "seed": seed,
                "tuning_params": cfg,
                "splits": {
                    "test": {
                        "accuracy": res["balanced_accuracy"],
                        "balanced_accuracy": res["balanced_accuracy"],
                        "macro_precision": res["balanced_accuracy"],
                        "macro_recall": res["balanced_accuracy"],
                        "macro_f1": res["macro_f1"],
                        "negative_log_likelihood": res["nll"],
                        "labels": res["targets"].tolist(),
                        "preds": res["preds"].tolist(),
                        "probabilities": res["probs"].tolist(),
                    }
                },
            },
        )


def _confirm_condition(
    cond: str,
    best_configs: dict[str, Any],
    test_ldr: torch.utils.data.DataLoader,
    device: torch.device,
    config: dict[str, Any],
    paths: dict[str, Path],
    n_cls: int,
) -> None:
    """Run confirmation training for every method within one imbalance condition."""
    train_ds = ImbalanceDataset(paths["data"] / f"manifest_{cond}.csv", device=device)
    for method in ["ce", "weighted_ce"]:
        _confirm_run(
            cond,
            method,
            best_configs.get(method, {"lr": 1e-3}),
            train_ds,
            test_ldr,
            device,
            config,
            paths,
            n_cls,
        )


def cmd_confirm(args: argparse.Namespace) -> None:
    """Train confirmation models."""
    config = load_config(args.config)
    paths = ensure_dirs(config)
    with (paths["data"] / "tuning_selections.json").open() as f:
        best_configs = json.load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_ds = ImbalanceDataset(
        paths["data"] / "manifest.csv", split_name="test", device=device
    )
    test_ldr = torch.utils.data.DataLoader(test_ds, batch_size=64)
    n_cls = test_ds.get_n_classes()
    for cond in ["balanced", "moderate"]:
        _confirm_condition(cond, best_configs, test_ldr, device, config, paths, n_cls)
