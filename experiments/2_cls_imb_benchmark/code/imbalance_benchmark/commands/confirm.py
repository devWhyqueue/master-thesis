from __future__ import annotations

import argparse
import json
from typing import Any

import torch

from imbalance_benchmark.modeling.workflows.confirmation import (
    RunContext,
    confirm_ce,
    confirm_crt,
    confirm_method,
    confirm_post_hoc,
)
from imbalance_benchmark.common import (
    bag_dataset_kwargs,
    ensure_dirs,
    load_config,
    split_paths,
    verify_signed_file,
)
from imbalance_benchmark.datasets.data import (
    TrainDataset,
    bag_collate,
)
from imbalance_benchmark.datasets.data import load_training_dataset
from imbalance_benchmark.manifest.freeze import verify_manifest_freeze
from imbalance_benchmark.manifest.seeds import derive_seed
from imbalance_benchmark.modeling.context import CONDITIONS, roster_for_regime

__all__ = ["cmd_confirm"]

CONFIRMATION_SEED_ROLES = [f"confirmation_initialization_{i}" for i in range(5)]


def _is_excluded(paths: dict[str, Any]) -> bool:
    """Return whether this split belongs to a recorded confirmatory exclusion."""
    return (paths["data"] / "confirmatory_exclusion.json").exists()


def _confirm_condition(
    cond: str,
    methods: tuple[str, ...],
    best_configs: dict[str, Any],
    run: RunContext,
) -> None:
    """Run confirmation training for every roster method within one imbalance condition."""
    file_name = (
        f"manifest_{cond}.csv"
        if cond in {"natural", "balanced"}
        else f"manifest_{run.assignment}_{cond}.csv"
    )
    train_ds: TrainDataset = load_training_dataset(
        run.paths["data"] / file_name,
        run.is_mil,
        device=run.device,
        bag_kwargs=run.bag_dataset_kwargs,
    )
    cond_configs = require_tuning_configs(best_configs, methods)
    ce_states: list[tuple[dict[str, Any], int]] | None = None
    for method in methods:
        cfg = cond_configs[method]
        if method == "ce":
            ce_states = confirm_ce(cond, cfg, train_ds, run)
        elif method == "post_hoc_logit_adjustment":
            assert ce_states is not None, (
                "CE must be confirmed before post-hoc adjustment"
            )
            confirm_post_hoc(cond, cfg, ce_states, train_ds, run)
        elif method == "crt":
            confirm_crt(cond, cfg, cond_configs["ce"], train_ds, run)
        else:
            confirm_method(cond, method, cfg, train_ds, run)


def _confirm_inputs(
    args: argparse.Namespace, paths: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load config, best tuning selections, and the locked-test loader for confirmation."""
    config = load_config(args.config)
    freeze_path = paths["data"] / "manifest_freeze.json"
    if not freeze_path.exists():
        raise FileNotFoundError("Run freeze successfully before confirmation")
    if freeze_path.exists():
        verify_manifest_freeze(json.loads(freeze_path.read_text()))
    condition = getattr(args, "condition", None)
    selection_name = (
        f"tuning_selections_{condition}.json" if condition else "tuning_selections.json"
    )
    selection_path = paths["data"] / selection_name
    verify_signed_file(selection_path)
    with selection_path.open() as f:
        best_configs = json.load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_mil = config.get("dataset", {}).get("regime", "patch") == "wsi"
    freeze = json.loads(freeze_path.read_text())
    class_names = list(freeze["class_names"])
    bag_kwargs = bag_dataset_kwargs(config, freeze) if is_mil else None
    test_ds = load_training_dataset(
        paths["data"] / "manifest.csv",
        is_mil,
        "test",
        device=device,
        class_names=class_names,
        bag_kwargs=bag_kwargs,
    )
    val_ds = load_training_dataset(
        paths["data"] / "manifest.csv",
        is_mil,
        "validation",
        device=device,
        class_names=class_names,
        bag_kwargs=bag_kwargs,
    )
    collate = bag_collate if is_mil else None
    val_ldr = torch.utils.data.DataLoader(val_ds, batch_size=64, collate_fn=collate)
    test_ldr = torch.utils.data.DataLoader(test_ds, batch_size=64, collate_fn=collate)
    seeds = [derive_seed(args.seed, role) for role in CONFIRMATION_SEED_ROLES]
    run_data = {
        "device": device,
        "config": config,
        "n_classes": len(class_names),
        "is_mil": is_mil,
        "val_loader": val_ldr,
        "test_loader": test_ldr,
        "paths": paths,
        "seeds": seeds,
        "class_names": class_names,
        "bag_dataset_kwargs": bag_kwargs or {},
    }
    return best_configs, run_data, freeze


def require_tuning_configs(
    configs: dict[str, Any], methods: tuple[str, ...]
) -> dict[str, Any]:
    """Refuse confirmation if a prespecified method lacks its signed selection."""
    missing = [
        method for method in methods if not isinstance(configs.get(method), dict)
    ]
    if missing:
        raise RuntimeError(f"missing tuning selection for methods: {missing}")
    return configs


def cmd_confirm(args: argparse.Namespace) -> None:
    """Fit every roster method's five confirmation seeds and emit locked test predictions."""
    if args.split_index is None:
        config = load_config(args.config)
        base_paths = ensure_dirs(config)
        if any(_is_excluded(split_paths(base_paths, index)) for index in range(3)):
            return
        for index in range(3):
            cmd_confirm(argparse.Namespace(**{**vars(args), "split_index": index}))
        return
    config = load_config(args.config)
    paths = split_paths(ensure_dirs(config), args.split_index)
    if _is_excluded(paths):
        return
    _confirm_split(args, paths)


def _confirm_split(args: argparse.Namespace, paths: dict[str, Any]) -> None:
    """Run every requested condition for one prepared patient split."""
    best_configs, run_data, freeze = _confirm_inputs(args, paths)
    methods = roster_for_regime(run_data["is_mil"])
    conditions = (args.condition,) if getattr(args, "condition", None) else CONDITIONS
    assignments = tuple(freeze.get("tail_assignments", {"native": []}))
    for cond in conditions:
        scoped_assignments = (
            ("unassigned",) if cond in {"natural", "balanced"} else assignments
        )
        for assignment in scoped_assignments:
            run = RunContext(**run_data, assignment=assignment)
            selected_assignment = "native" if assignment == "unassigned" else assignment
            selected = best_configs.get(selected_assignment, {}).get(cond, {})
            _confirm_condition(cond, methods, selected, run)
