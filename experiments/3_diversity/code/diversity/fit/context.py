"""Per-split run data (locked loaders, seeds) and per-allocation tuning selections."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from imbalance_benchmark.common import verify_signed_file
from imbalance_benchmark.datasets.data import load_training_dataset
from imbalance_benchmark.manifest.freeze import verify_manifest_freeze
from imbalance_benchmark.modeling.training import build_evaluation_loader

from diversity.manifests import ANCHOR_ASSIGNMENT

__all__ = ["fit_run_data", "load_selection", "method_config"]

# Mirrors commands.confirm.CONFIRMATION_SEED_ROLES without importing that
# CLI-oriented module: this is the fixed, frozen list of seed-role names
# every exp-2 freeze already carries under ``seed_roles``.
CONFIRMATION_SEED_ROLES = [f"confirmation_initialization_{i}" for i in range(5)]


def _load_verified_freeze(exp3_paths: dict[str, Path]) -> dict[str, Any]:
    freeze_path = exp3_paths["data"] / "manifest_freeze.json"
    verify_signed_file(freeze_path)
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    verify_manifest_freeze(freeze)
    return freeze


def _locked_loaders(manifest: Path, class_names: list[str]) -> tuple[Any, Any]:
    test_ds = load_training_dataset(manifest, False, "test", class_names=class_names)
    val_ds = load_training_dataset(
        manifest, False, "validation", class_names=class_names
    )
    return build_evaluation_loader(val_ds, False), build_evaluation_loader(
        test_ds, False
    )


def fit_run_data(exp3_paths: dict[str, Path]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load exp-3's derived freeze and the locked val/test loaders it shares across units."""
    freeze = _load_verified_freeze(exp3_paths)
    class_names = list(freeze["class_names"])
    val_loader, test_loader = _locked_loaders(
        exp3_paths["data"] / "manifest.csv", class_names
    )
    run_data = {
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "config": freeze["runtime_config"],
        "n_classes": len(class_names),
        "is_mil": False,
        "class_names": class_names,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "paths": exp3_paths,
        "seeds": [int(freeze["seed_roles"][role]) for role in CONFIRMATION_SEED_ROLES],
        "exposure_budgets": freeze["exposure_budgets"],
        "feature_provenance": freeze.get("feature_provenance"),
        "difficulty": freeze.get("difficulty_evidence", {}).get("difficulty", {}),
    }
    return run_data, freeze


def load_selection(exp2_data_dir: Path, allocation: str) -> dict[str, Any]:
    """Load and verify exp-2's signed tuning selections for one allocation."""
    path = exp2_data_dir / f"tuning_selections_{allocation}.json"
    verify_signed_file(path)
    return json.loads(path.read_text(encoding="utf-8"))


def method_config(
    selections: dict[str, Any], allocation: str, method: str
) -> dict[str, Any]:
    """The random cell's tuned config, inherited unchanged by narrow and wide (plan Stage 3)."""
    anchor = ANCHOR_ASSIGNMENT[allocation]
    return selections[anchor][allocation][method]
