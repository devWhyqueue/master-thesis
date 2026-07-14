from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from imbalance_benchmark.common import (
    bag_dataset_kwargs,
    ensure_dirs,
    load_config,
    sign_file,
    split_paths,
    write_json,
)
from imbalance_benchmark.datasets.data import (
    bag_collate,
)
from imbalance_benchmark.datasets.data import load_training_dataset
from imbalance_benchmark.manifest.freeze import verify_manifest_freeze
from imbalance_benchmark.manifest.seeds import derive_seed
from imbalance_benchmark.modeling.context import CONDITIONS, Regime, roster_for_regime
from imbalance_benchmark.modeling.workflows.tuning_aggregate import (
    TuningScope,
    tune_across_splits,
)

__all__ = ["cmd_tune"]


def _is_excluded(paths: dict[str, Path]) -> bool:
    """Return whether a failed pilot/freeze excludes this confirmatory workflow."""
    return (paths["data"] / "confirmatory_exclusion.json").exists()


def _tuning_inputs(
    args: argparse.Namespace, paths: dict[str, Path]
) -> tuple[dict[str, Path], Regime, torch.utils.data.DataLoader]:
    """Load config, the natural-validation loader, and the regime for the tuning sweep."""
    config = load_config(args.config)
    freeze_path = paths["data"] / "manifest_freeze.json"
    freeze = json.loads(freeze_path.read_text())
    verify_manifest_freeze(freeze)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_mil = config.get("dataset", {}).get("regime", "patch") == "wsi"
    bag_kwargs = bag_dataset_kwargs(config, freeze) if is_mil else None
    val_ds = load_training_dataset(
        paths["data"] / "manifest.csv",
        is_mil,
        "validation",
        device=device,
        class_names=list(freeze["class_names"]),
        bag_kwargs=bag_kwargs,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=64, collate_fn=bag_collate if is_mil else None
    )
    regime = Regime(
        device,
        config,
        val_ds.get_n_classes(),
        is_mil,
        locked_class_names=list(freeze["class_names"]),
        bag_dataset_kwargs=bag_kwargs or {},
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


def _tuning_seeds(seed: int) -> list[int]:
    """Return the two locked initialization seeds used for every candidate."""
    return [derive_seed(seed, f"tuning_initialization_{index}") for index in range(2)]


def _conditions(args: argparse.Namespace) -> tuple[str, ...]:
    """Return the requested condition scope, defaulting to the whole roster."""
    return (args.condition,) if getattr(args, "condition", None) else CONDITIONS


def _output_name(args: argparse.Namespace) -> str:
    """Return the condition-safe tuning selection output name."""
    return (
        f"tuning_selections_{args.condition}.json"
        if getattr(args, "condition", None)
        else "tuning_selections.json"
    )


def _manifest_name(condition: str, assignment: str) -> str:
    """Resolve one condition/assignment's frozen training manifest name."""
    return (
        f"manifest_{condition}.csv"
        if condition in {"natural", "balanced"}
        else f"manifest_{assignment}_{condition}.csv"
    )


def _tune_all_splits(args: argparse.Namespace, started: float) -> None:
    """Tune one configuration against the equal-weight three-split objective."""
    base_paths = ensure_dirs(load_config(args.config))
    scopes = [
        _tuning_inputs(args, split_paths(base_paths, index)) for index in range(3)
    ]
    if any(_is_excluded(paths) for paths, _, _ in scopes):
        return
    selections = _combined_selections(scopes, args)
    for paths, _, _ in scopes:
        selection_path = paths["data"] / _output_name(args)
        write_json(selection_path, selections)
        sign_file(selection_path)
        _write_tuning_cost(paths, started, getattr(args, "condition", None))


def _combined_selections(
    scopes: list[tuple[dict[str, Path], Regime, torch.utils.data.DataLoader]],
    args: argparse.Namespace,
) -> dict[str, dict[str, Any]]:
    """Fit the shared configuration-selection objective across all three splits."""
    paths, regime, _ = scopes[0]
    freeze = json.loads((paths["data"] / "manifest_freeze.json").read_text())
    assignments = tuple(freeze.get("tail_assignments", {"native": []}))
    selections: dict[str, dict[str, Any]] = {
        assignment: {} for assignment in assignments
    }
    for condition in _conditions(args):
        scoped = ("native",) if condition in {"natural", "balanced"} else assignments
        selected = tune_across_splits(
            roster_for_regime(regime.is_mil),
            _combined_scopes(scopes, condition, scoped),
            _tuning_seeds(args.seed),
        )
        for assignment in scoped:
            selections[assignment][condition] = selected
    return selections


def _combined_scopes(
    scopes: list[tuple[dict[str, Path], Regime, torch.utils.data.DataLoader]],
    condition: str,
    assignments: tuple[str, ...],
) -> list[TuningScope]:
    """Build the split/assignment training datasets for one shared selection."""
    result = []
    for assignment in assignments:
        for paths, regime, loader in scopes:
            result.append(
                TuningScope(
                    regime,
                    loader,
                    load_training_dataset(
                        paths["data"] / _manifest_name(condition, assignment),
                        regime.is_mil,
                        device=regime.device,
                        class_names=regime.locked_class_names,
                        bag_kwargs=regime.bag_dataset_kwargs,
                    ),
                )
            )
    return result


def _write_tuning_cost(
    paths: dict[str, Path], started: float, condition: str | None = None
) -> None:
    """Persist validation-search cost separately from locked confirmation fits."""
    elapsed = time.perf_counter() - started
    write_json(
        paths["data"]
        / (
            f"tuning_search_cost_{condition}.json"
            if condition
            else "tuning_search_cost.json"
        ),
        {
            "wall_clock_seconds": elapsed,
            "accelerator_hours": elapsed / 3600 if torch.cuda.is_available() else 0.0,
            "peak_accelerator_memory_bytes": int(torch.cuda.max_memory_allocated())
            if torch.cuda.is_available()
            else 0,
        },
    )
