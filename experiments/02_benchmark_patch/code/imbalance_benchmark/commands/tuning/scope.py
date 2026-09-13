from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from imbalance_benchmark.common import (
    compute_sha256,
    ensure_dirs,
    load_config,
    split_paths,
)
from imbalance_benchmark.datasets.data import load_training_dataset
from imbalance_benchmark.manifest.freeze import verify_manifest_freeze
from imbalance_benchmark.modeling.context import INPUT_DIM, Regime
from imbalance_benchmark.modeling.training import build_evaluation_loader
from imbalance_benchmark.modeling.workflows.tuning.aggregation.aggregate import (
    TuningScope,
)
from imbalance_benchmark.modeling.workflows.tuning.aggregation.tuning_budget import (
    tuning_example_budget,
)
from imbalance_benchmark.modeling.workflows.tuning.tuning_schedule import _manifest_name

__all__ = [
    "bank_bytes_for",
    "load_shard_scope",
]

# On-disk feature-bank dtype -> bytes per element (feature_extraction.dtype).
_BANK_ITEMSIZE = {"float16": 2, "float32": 4}


def _is_excluded(paths: dict[str, Path]) -> bool:
    return (paths["data"] / "confirmatory_exclusion.json").exists()


def _tuning_seeds(freeze: dict[str, Any]) -> list[int]:
    roles = freeze.get("seed_roles", {})
    return [int(roles[f"tuning_initialization_{index}"]) for index in range(2)]


def _tuning_inputs(
    args: argparse.Namespace, paths: dict[str, Path], capacity_hint: int | None = None
) -> tuple[dict[str, Path], Regime, torch.utils.data.DataLoader]:
    """Load config, the natural-validation loader, and the regime for the tuning sweep."""
    freeze_path = paths["data"] / "manifest_freeze.json"
    freeze = json.loads(freeze_path.read_text())
    verify_manifest_freeze(freeze)
    config = freeze["runtime_config"]
    is_mil = config.get("dataset", {}).get("regime", "patch") == "wsi"
    val_ds = load_training_dataset(
        paths["data"] / "manifest.csv",
        is_mil,
        "validation",
        class_names=list(freeze["class_names"]),
        capacity_hint=capacity_hint,
    )
    return (
        paths,
        Regime(
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            config,
            val_ds.get_n_classes(),
            is_mil,
            locked_class_names=list(freeze["class_names"]),
            method_grids=freeze.get("method_grids", {}),
            exposure_budgets=freeze.get("exposure_budgets", {}),
            difficulty=freeze.get("difficulty_evidence", {}).get("difficulty", {}),
        ),
        build_evaluation_loader(val_ds, is_mil),
    )


def _frozen_shard_context(
    args: argparse.Namespace,
    load_scopes: bool = True,
) -> tuple[
    dict[str, Path],
    list[tuple[dict[str, Path], Regime, torch.utils.data.DataLoader]],
    dict[str, Any],
    list[str],
    list[set[str]],
]:
    base = ensure_dirs(load_config(args.config))
    paths = [
        split_paths(base, index)["data"] / "manifest_freeze.json" for index in range(3)
    ]
    freezes = [json.loads(path.read_text()) for path in paths]
    for freeze in freezes:
        verify_manifest_freeze(freeze)
    scopes = (
        [_tuning_inputs(args, split_paths(base, index)) for index in range(3)]
        if load_scopes
        else []
    )
    fingerprint = [compute_sha256(path) for path in paths]
    accepted = [
        {current, *split_freeze.get("superseded_freeze_file_hashes", [])}
        for current, split_freeze in zip(fingerprint, freezes)
    ]
    return (base, scopes, freezes[0], fingerprint, accepted)


def _bank_capacity_for(
    base: dict[str, Path], condition: str, assignment: str, split_index: int
) -> tuple[dict[str, Path], Path, int]:
    """Manifest path and row count one fit's bank must hold (validation + condition rows)."""
    paths = split_paths(base, split_index)
    manifest = paths["data"] / _manifest_name(condition, assignment)
    frame = pd.read_csv(paths["data"] / "manifest.csv")
    capacity = int((frame["split"] == "validation").sum()) + len(pd.read_csv(manifest))
    return paths, manifest, capacity


def bank_bytes_for(
    base: dict[str, Path],
    condition: str,
    assignment: str,
    dtype: str,
    split_index: int = 0,
) -> int:
    """Estimate one fit's on-disk feature-bank footprint before it loads.

    Reuses ``_bank_capacity_for``'s row count so the runtime-K packing cap
    (shard_workers._vram_capped_workers) can never drift from the bank.
    """
    _, _, capacity = _bank_capacity_for(base, condition, assignment, split_index)
    return capacity * INPUT_DIM * _BANK_ITEMSIZE.get(dtype, 4)


def load_shard_scope(
    args: argparse.Namespace,
    base: dict[str, Path],
    condition: str,
    assignment: str,
    split_index: int,
    scope_index: int,
    cost_records: list[dict[str, int]],
) -> TuningScope:
    """Load one tuning observation's validation and training data into one bank."""
    paths, manifest, capacity = _bank_capacity_for(
        base, condition, assignment, split_index
    )
    paths, regime, loader = _tuning_inputs(args, paths, capacity)
    return TuningScope(
        regime,
        loader,
        load_training_dataset(
            manifest,
            regime.is_mil,
            class_names=regime.locked_class_names,
            capacity_hint=capacity,
        ),
        cost_records,
        tuning_example_budget(regime, condition),
        assignment,
        split_index,
        scope_index,
    )
