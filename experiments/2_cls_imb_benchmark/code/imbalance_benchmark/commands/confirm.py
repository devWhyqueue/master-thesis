from __future__ import annotations

import argparse
import json
from typing import Any

import torch

from imbalance_benchmark.commands.confirm_methods import (
    RunContext,
    confirm_ce,
    confirm_crt,
    confirm_method,
    confirm_post_hoc,
)
from imbalance_benchmark.common import ensure_dirs, load_config
from imbalance_benchmark.data import (
    BagFeatureDataset,
    ImbalanceDataset,
    TrainDataset,
    bag_collate,
)
from imbalance_benchmark.manifest.freeze import verify_manifest_freeze
from imbalance_benchmark.manifest.seeds import derive_seed
from imbalance_benchmark.modeling.context import CONDITIONS, roster_for_regime

__all__ = ["cmd_confirm"]

CONFIRMATION_SEED_ROLES = [f"confirmation_initialization_{i}" for i in range(5)]


def _confirm_condition(
    cond: str, methods: tuple[str, ...], best_configs: dict[str, Any], run: RunContext
) -> None:
    """Run confirmation training for every roster method within one imbalance condition."""
    dataset_cls = BagFeatureDataset if run.is_mil else ImbalanceDataset
    train_ds: TrainDataset = dataset_cls(
        run.paths["data"] / f"manifest_{cond}.csv", device=run.device
    )
    cond_configs = best_configs.get(cond, {})
    ce_states: list[dict[str, Any]] | None = None
    for method in methods:
        cfg = cond_configs.get(method, {"lr": 1e-3})
        if method == "ce":
            ce_states = confirm_ce(cond, cfg, train_ds, run)
        elif method == "post_hoc_logit_adjustment":
            assert ce_states is not None, (
                "CE must be confirmed before post-hoc adjustment"
            )
            confirm_post_hoc(cond, cfg, ce_states, train_ds, run)
        elif method == "crt":
            confirm_crt(cond, cfg, cond_configs.get("ce", {"lr": 1e-3}), train_ds, run)
        else:
            confirm_method(cond, method, cfg, train_ds, run)


def _confirm_inputs(args: argparse.Namespace) -> tuple[dict[str, Any], RunContext]:
    """Load config, best tuning selections, and the locked-test loader for confirmation."""
    config = load_config(args.config)
    paths = ensure_dirs(config)
    freeze_path = paths["data"] / "manifest_freeze.json"
    if freeze_path.exists():
        verify_manifest_freeze(json.loads(freeze_path.read_text()))
    with (paths["data"] / "tuning_selections.json").open() as f:
        best_configs = json.load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_mil = config.get("dataset", {}).get("regime", "patch") == "wsi"
    dataset_cls = BagFeatureDataset if is_mil else ImbalanceDataset
    test_ds = dataset_cls(
        paths["data"] / "manifest.csv", split_name="test", device=device
    )
    val_ds = dataset_cls(
        paths["data"] / "manifest.csv", split_name="validation", device=device
    )
    collate = bag_collate if is_mil else None
    val_ldr = torch.utils.data.DataLoader(val_ds, batch_size=64, collate_fn=collate)
    test_ldr = torch.utils.data.DataLoader(test_ds, batch_size=64, collate_fn=collate)
    seeds = [derive_seed(args.seed, role) for role in CONFIRMATION_SEED_ROLES]
    run = RunContext(
        device,
        config,
        test_ds.get_n_classes(),
        is_mil,
        val_ldr,
        test_ldr,
        paths,
        seeds,
        test_ds.classes,
    )
    return best_configs, run


def cmd_confirm(args: argparse.Namespace) -> None:
    """Fit every roster method's five confirmation seeds and emit locked test predictions."""
    best_configs, run = _confirm_inputs(args)
    methods = roster_for_regime(run.is_mil)
    for cond in CONDITIONS:
        _confirm_condition(cond, methods, best_configs, run)
