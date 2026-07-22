from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from imbalance_benchmark.common import (
    bag_dataset_kwargs,
    compute_sha256,
    ensure_dirs,
    load_config,
    sign_file,
    split_paths,
    write_json,
)
from imbalance_benchmark.datasets.data import load_training_dataset
from imbalance_benchmark.manifest.freeze import verify_manifest_freeze
from imbalance_benchmark.modeling.context import CONDITIONS, Regime, roster_for_regime
from imbalance_benchmark.modeling.training import build_evaluation_loader
from imbalance_benchmark.modeling.workflows.tuning_aggregate import (
    summarize_tuning_cost,
    tune_across_splits,
)
from imbalance_benchmark.modeling.workflows.tuning.tuning_reduction import (
    condition_is_reusable,
    write_base_selections,
    write_final_selections,
    write_serial_cost,
)
from imbalance_benchmark.modeling.workflows.tuning.tuning_artifacts import selected_ce
from imbalance_benchmark.modeling.workflows.tuning.tuning_shards import (
    ShardSpec,
    combined_scopes,
    run_candidate_shard,
)
from imbalance_benchmark.modeling.workflows.tuning.tuning_bundle import run_shard_bundle
from imbalance_benchmark.modeling.workflows.tuning.tuning_schedule import (
    array_coordinates,
    phase_methods,
    requested_shard,
)


def _is_excluded(paths: dict[str, Path]) -> bool:
    return (paths["data"] / "confirmatory_exclusion.json").exists()


def _tuning_inputs(
    args: argparse.Namespace, paths: dict[str, Path]
) -> tuple[dict[str, Path], Regime, torch.utils.data.DataLoader]:
    """Load config, the natural-validation loader, and the regime for the tuning sweep."""
    config = load_config(args.config)
    freeze_path = paths["data"] / "manifest_freeze.json"
    freeze = json.loads(freeze_path.read_text())
    verify_manifest_freeze(freeze)
    config = freeze["runtime_config"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_mil = config.get("dataset", {}).get("regime", "patch") == "wsi"
    bag_kwargs = bag_dataset_kwargs(config, freeze) if is_mil else None
    val_ds = load_training_dataset(
        paths["data"] / "manifest.csv",
        is_mil,
        "validation",
        class_names=list(freeze["class_names"]),
        bag_kwargs=bag_kwargs,
    )
    val_loader = build_evaluation_loader(val_ds, is_mil)
    regime = Regime(
        device,
        config,
        val_ds.get_n_classes(),
        is_mil,
        locked_class_names=list(freeze["class_names"]),
        bag_dataset_kwargs=bag_kwargs or {},
        method_grids=freeze.get("method_grids", {}),
        update_budgets=freeze.get("update_budgets", {}),
    )
    return paths, regime, val_loader


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
            roster_for_regime(regime.is_mil),
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
    base_paths = ensure_dirs(load_config(args.config))
    raw_scopes = [
        _tuning_inputs(args, split_paths(base_paths, index)) for index in range(3)
    ]
    freeze_paths = [scope[0]["data"] / "manifest_freeze.json" for scope in raw_scopes]
    freeze = json.loads(freeze_paths[0].read_text())
    return (
        base_paths,
        raw_scopes,
        freeze,
        [compute_sha256(path) for path in freeze_paths],
    )


def cmd_tune_shard(args: argparse.Namespace) -> None:
    """Run one resumable frozen-candidate shard."""
    if run_shard_bundle(args):
        return
    base, raw_scopes, freeze, fingerprint = _frozen_shard_context(args)
    if any(_is_excluded(paths) for paths, _, _ in raw_scopes):
        return
    shard_index, observation_index = array_coordinates(
        args.shard_index,
        args.observation_index,
        args.observations_per_candidate,
        args.shard_offset,
    )
    spec = requested_shard(
        shard_index,
        args.phase,
        args.group,
        raw_scopes[0][1].is_mil,
        freeze["method_grids"],
        observation_index,
    )
    if spec is None:
        return
    _run_shard(base, raw_scopes, freeze, fingerprint, spec)


def _run_shard(
    base: dict[str, Path],
    raw_scopes: list[tuple[dict[str, Path], Regime, torch.utils.data.DataLoader]],
    freeze: dict[str, Any],
    fingerprint: list[str],
    spec: ShardSpec,
) -> None:
    assignments = tuple(freeze.get("tail_assignments", {"native": []}))
    if condition_is_reusable(
        base, spec.condition, roster_for_regime(raw_scopes[0][1].is_mil), assignments
    ):
        return
    scoped = ("native",) if spec.condition in {"natural", "balanced"} else assignments
    records: list[dict[str, int]] = []
    scopes = combined_scopes(raw_scopes, spec.condition, scoped, records)
    run_candidate_shard(
        spec,
        scopes,
        _tuning_seeds(freeze),
        fingerprint,
        base["data"],
        selected_ce(base["data"], spec.condition)
        if spec.phase == "dependent"
        else None,
    )


def cmd_tune_reduce(args: argparse.Namespace) -> None:
    """Reduce complete shards into the benchmark's signed selection interface."""
    base, raw_scopes, freeze, fingerprint = _frozen_shard_context(args)
    is_mil = raw_scopes[0][1].is_mil
    base_methods = phase_methods(is_mil, "base")
    if args.phase == "base":
        write_base_selections(
            base,
            freeze,
            fingerprint,
            base_methods,
            roster_for_regime(is_mil),
            CONDITIONS,
        )
        return
    write_final_selections(
        base,
        freeze,
        fingerprint,
        base_methods,
        phase_methods(is_mil, "dependent"),
        CONDITIONS,
    )
