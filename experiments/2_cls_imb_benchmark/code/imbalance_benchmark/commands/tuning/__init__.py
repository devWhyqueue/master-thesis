from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from imbalance_benchmark.common import (
    compute_sha256,
    ensure_dirs,
    load_config,
    sign_file,
    split_paths,
    write_json,
)
from imbalance_benchmark.datasets.data import load_training_dataset
from imbalance_benchmark.manifest.freeze import verify_manifest_freeze
from imbalance_benchmark.modeling.context import (
    CONDITIONS,
    Regime,
    roster_for_condition,
)
from imbalance_benchmark.modeling.training import build_evaluation_loader
from imbalance_benchmark.modeling.workflows.tuning_aggregate import (
    summarize_tuning_cost,
    tune_across_splits,
)
from imbalance_benchmark.modeling.workflows.tuning.tuning_execution import (
    reduce_tuning_shards,
)
from imbalance_benchmark.modeling.workflows.tuning.tuning_shards import combined_scopes

__all__ = ["cmd_tune", "cmd_tune_reduce"]


def _is_excluded(paths: dict[str, Path]) -> bool:
    return (paths["data"] / "confirmatory_exclusion.json").exists()


def write_serial_cost(
    paths: dict[str, Path],
    started: float,
    search_cost: dict[str, float | int],
    condition: str | None,
) -> None:
    """Preserve cost output for the legacy serial tuning command."""
    elapsed = time.perf_counter() - started
    name = (
        f"tuning_search_cost_{condition}.json"
        if condition
        else "tuning_search_cost.json"
    )
    write_json(
        paths["data"] / name,
        {
            "wall_clock_seconds": elapsed,
            "accelerator_hours": elapsed / 3600 if torch.cuda.is_available() else 0.0,
            "peak_accelerator_memory_bytes": int(torch.cuda.max_memory_allocated())
            if torch.cuda.is_available()
            else 0,
            **search_cost,
        },
    )


def _tuning_inputs(
    args: argparse.Namespace, paths: dict[str, Path]
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
            update_budgets=freeze.get("update_budgets", {}),
        ),
        build_evaluation_loader(val_ds, is_mil),
    )


def cmd_tune(args: argparse.Namespace) -> None:
    """Run the validation-only hyperparameter search for every roster method and condition."""
    started = time.perf_counter()
    if args.split_index is not None:
        raise ValueError(
            "Definitive tuning selects one configuration across all three patient splits; "
            "do not pass --split-index."
        )
    _tune_all_splits(args, started)


def _tuning_seeds(freeze: dict[str, Any]) -> list[int]:
    roles = freeze.get("seed_roles", {})
    return [int(roles[f"tuning_initialization_{index}"]) for index in range(2)]


def _conditions(args: argparse.Namespace) -> tuple[str, ...]:
    return (args.condition,) if getattr(args, "condition", None) else CONDITIONS


def _output_name(args: argparse.Namespace) -> str:
    return (
        f"tuning_selections_{args.condition}.json"
        if getattr(args, "condition", None)
        else "tuning_selections.json"
    )


def _tune_all_splits(args: argparse.Namespace, started: float) -> None:
    """Tune one configuration against the equal-weight three-split objective."""
    base_paths = ensure_dirs(load_config(args.config))
    scopes = [
        _tuning_inputs(args, split_paths(base_paths, index)) for index in range(3)
    ]
    if any(_is_excluded(paths) for paths, _, _ in scopes):
        return
    selections, search_cost = _combined_selections(scopes, args)
    for paths, _, _ in scopes:
        selection_path = paths["data"] / _output_name(args)
        write_json(selection_path, selections)
        sign_file(selection_path)
        write_serial_cost(paths, started, search_cost, getattr(args, "condition", None))


def _combined_selections(
    scopes: list[tuple[dict[str, Path], Regime, torch.utils.data.DataLoader]],
    args: argparse.Namespace,
) -> tuple[dict[str, dict[str, Any]], dict[str, float | int]]:
    """Fit the shared configuration-selection objective across all three splits."""
    paths, regime, _ = scopes[0]
    freeze = json.loads((paths["data"] / "manifest_freeze.json").read_text())
    assignments = tuple(freeze.get("tail_assignments", {"native": []}))
    selections: dict[str, dict[str, Any]] = {
        assignment: {} for assignment in assignments
    }
    cost_records: list[dict[str, int]] = []
    for condition in _conditions(args):
        scoped = ("native",) if condition in {"natural", "balanced"} else assignments
        selected = tune_across_splits(
            roster_for_condition(regime.is_mil, condition),
            combined_scopes(scopes, condition, scoped, cost_records),
            _tuning_seeds(freeze),
        )
        for assignment in scoped:
            selections[assignment][condition] = selected
    return selections, summarize_tuning_cost(cost_records)


def _frozen_shard_context(
    args: argparse.Namespace,
) -> tuple[
    dict[str, Path],
    list[tuple[dict[str, Path], Regime, torch.utils.data.DataLoader]],
    dict[str, Any],
    list[str],
]:
    base = ensure_dirs(load_config(args.config))
    scopes = [_tuning_inputs(args, split_paths(base, index)) for index in range(3)]
    paths = [scope[0]["data"] / "manifest_freeze.json" for scope in scopes]
    return (
        base,
        scopes,
        json.loads(paths[0].read_text()),
        [compute_sha256(path) for path in paths],
    )


def cmd_tune_reduce(args: argparse.Namespace) -> None:
    """Reduce complete shards into the benchmark's signed selection interface."""
    base, raw_scopes, freeze, fingerprint = _frozen_shard_context(args)
    reduce_tuning_shards(
        base,
        raw_scopes,
        freeze,
        fingerprint,
        args.phase,
        getattr(args, "condition", None),
    )
