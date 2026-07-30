from __future__ import annotations

import argparse
import json
from pathlib import Path
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
    ensure_dirs,
    load_config,
    split_paths,
    verify_signed_file,
)
from imbalance_benchmark.modeling.workflows.tuning.candidate_registry import (
    tuning_locked,
)
from imbalance_benchmark.datasets.data import TrainDataset
from imbalance_benchmark.datasets.data import load_training_dataset
from imbalance_benchmark.manifest.freeze import verify_manifest_freeze
from imbalance_benchmark.modeling.context import CONDITIONS, roster_for_regime
from imbalance_benchmark.modeling.training import build_evaluation_loader

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
        run.paths["data"] / file_name, run.is_mil, class_names=run.class_names
    )
    cond_configs = require_tuning_configs(
        run.paths["data"].parent.parent / "data", cond, best_configs, methods
    )
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


def _confirm_run_data(paths: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load freeze and the locked val/test loaders shared by every condition."""
    freeze_path = paths["data"] / "manifest_freeze.json"
    if not freeze_path.exists():
        raise FileNotFoundError("Run freeze successfully before confirmation")
    verify_manifest_freeze(json.loads(freeze_path.read_text()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    freeze = json.loads(freeze_path.read_text())
    config = freeze["runtime_config"]
    is_mil = config.get("dataset", {}).get("regime", "patch") == "wsi"
    class_names = list(freeze["class_names"])
    test_ds = load_training_dataset(
        paths["data"] / "manifest.csv", is_mil, "test", class_names=class_names
    )
    val_ds = load_training_dataset(
        paths["data"] / "manifest.csv", is_mil, "validation", class_names=class_names
    )
    val_ldr = build_evaluation_loader(val_ds, is_mil)
    test_ldr = build_evaluation_loader(test_ds, is_mil)
    seed_roles = freeze.get("seed_roles", {})
    seeds = [int(seed_roles[role]) for role in CONFIRMATION_SEED_ROLES]
    run_data = {
        "device": device,
        "config": config,
        "n_classes": len(class_names),
        "is_mil": is_mil,
        "class_names": class_names,
        "val_loader": val_ldr,
        "test_loader": test_ldr,
        "paths": paths,
        "seeds": seeds,
        "update_budgets": freeze["update_budgets"],
        "feature_provenance": freeze.get("feature_provenance"),
    }
    return run_data, freeze


def _load_selections(paths: dict[str, Any], name: str) -> dict[str, Any]:
    """Load and verify one signed tuning-selection interface file."""
    selection_path = paths["data"] / name
    verify_signed_file(selection_path)
    with selection_path.open() as f:
        return json.load(f)


def _load_condition_selections(paths: dict[str, Any], condition: str) -> dict[str, Any]:
    """Load one condition's signed selections, as written by ``tune-final-reduce``."""
    return _load_selections(paths, f"tuning_selections_{condition}.json")


def _confirm_inputs(
    args: argparse.Namespace, paths: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load config, best tuning selections, and the locked-test loader for confirmation."""
    run_data, freeze = _confirm_run_data(paths)
    condition = getattr(args, "condition", None)
    name = (
        f"tuning_selections_{condition}.json" if condition else "tuning_selections.json"
    )
    best_configs = _load_selections(paths, name)
    return best_configs, run_data, freeze


def require_tuning_configs(
    root: Path, condition: str, configs: dict[str, Any], methods: tuple[str, ...]
) -> dict[str, Any]:
    """Refuse confirmation if a method lacks its selection or its tuning lock.

    A selection can exist yet still be mid-search: an adaptive round's
    winner may be an unresolved edge case awaiting another round. Only a
    signed tuning-round-state showing every method resolved or correctly
    marked tuning-limited - the tuning lock - may unblock confirmation.
    """
    missing = [
        method for method in methods if not isinstance(configs.get(method), dict)
    ]
    if missing:
        raise RuntimeError(f"missing tuning selection for methods: {missing}")
    if not tuning_locked(root, condition, methods):
        raise RuntimeError(
            f"tuning lock unresolved for condition {condition!r}; "
            "confirmation refused until every search window is resolved "
            "or correctly marked tuning-limited"
        )
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
