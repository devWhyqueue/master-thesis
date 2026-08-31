from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
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
    scoped_assignments,
)
from imbalance_benchmark.modeling.training import build_evaluation_loader
from imbalance_benchmark.modeling.workflows.tuning.aggregation.aggregate import (
    TuningScope,
    summarize_tuning_cost,
    tune_across_splits,
)
from imbalance_benchmark.modeling.workflows.tuning.aggregation.candidate_registry import (
    merge_round_state,
)
from imbalance_benchmark.modeling.workflows.tuning.tuning_execution import (
    reduce_tuning_shards,
)
from imbalance_benchmark.modeling.workflows.tuning.aggregation.tuning_budget import (
    tuning_example_budget,
)
from imbalance_benchmark.modeling.workflows.tuning.tuning_schedule import (
    _manifest_name,
    combined_scopes,
)

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
    base = ensure_dirs(load_config(args.config))
    scopes = [_tuning_inputs(args, split_paths(base, index)) for index in range(3)]
    if any(_is_excluded(paths) for paths, _, _ in scopes):
        return
    selections, search_cost = _combined_selections(base["data"], scopes, args)
    for paths, _, _ in scopes:
        selection_path = paths["data"] / _output_name(args)
        write_json(selection_path, selections)
        sign_file(selection_path)
        write_serial_cost(paths, started, search_cost, getattr(args, "condition", None))


_RESOLVED_ROUND_STATE = {
    "resolved": True,
    "tuning_limited": False,
    "lr_window": None,
    "next_lr_window": None,
    "strength_window": None,
    "next_strength_window": None,
}


def _lock_condition(root: Path, condition: str, methods: tuple[str, ...]) -> None:
    """Sign this condition's tuning lock so confirmation accepts the legacy path's result.

    This serial command sweeps each method's *entire* configured grid in one
    shot rather than the sharded wave pipeline's adaptive round-by-round
    narrowing, so every method's result is genuinely final -- there is no
    partial window awaiting another round, hence unconditionally "resolved"
    (never "tuning_limited", which implies a budget cutoff short of that).
    """
    merge_round_state(
        root, condition, {method: dict(_RESOLVED_ROUND_STATE) for method in methods}
    )


def _combined_selections(
    root: Path,
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
        if not (scoped := scoped_assignments(condition, freeze, assignments)):
            continue  # not constructed for this dataset (plans/03,04)
        methods = roster_for_condition(regime.is_mil, condition)
        selected = tune_across_splits(
            methods,
            combined_scopes(scopes, condition, scoped, cost_records),
            _tuning_seeds(freeze),
        )
        for assignment in scoped:
            selections[assignment][condition] = selected
        _lock_condition(root, condition, methods)
    return selections, summarize_tuning_cost(cost_records)


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
    paths = split_paths(base, split_index)
    manifest = paths["data"] / _manifest_name(condition, assignment)
    frame = pd.read_csv(paths["data"] / "manifest.csv")
    capacity = int((frame["split"] == "validation").sum()) + len(pd.read_csv(manifest))
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


def cmd_tune_reduce(args: argparse.Namespace) -> None:
    """Reduce complete shards into the benchmark's signed selection interface."""
    base, _, freeze, fingerprint, accepted = _frozen_shard_context(args, False)
    reduce_tuning_shards(
        base,
        freeze,
        fingerprint,
        args.phase,
        getattr(args, "condition", None),
        accepted,
    )
