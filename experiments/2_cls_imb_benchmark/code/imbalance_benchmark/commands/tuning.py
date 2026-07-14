from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from imbalance_benchmark.common import ensure_dirs, load_config, split_paths, write_json
from imbalance_benchmark.datasets.data import (
    BagFeatureDataset,
    ImbalanceDataset,
    bag_collate,
)
from imbalance_benchmark.manifest.freeze import verify_manifest_freeze
from imbalance_benchmark.manifest.seeds import derive_seed
from imbalance_benchmark.modeling.context import CONDITIONS, Regime, roster_for_regime
from imbalance_benchmark.modeling.workflows.tuning_aggregate import (
    TuningScope,
    tune_across_splits,
)
from imbalance_benchmark.modeling.workflows.tuning_search import tune_condition

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
    if freeze_path.exists():
        verify_manifest_freeze(json.loads(freeze_path.read_text()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_mil = config.get("dataset", {}).get("regime", "patch") == "wsi"
    dataset_cls = BagFeatureDataset if is_mil else ImbalanceDataset
    val_ds = dataset_cls(
        paths["data"] / "manifest.csv", split_name="validation", device=device
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=64, collate_fn=bag_collate if is_mil else None
    )
    regime = Regime(device, config, val_ds.get_n_classes(), is_mil)
    return paths, regime, val_loader


def cmd_tune(args: argparse.Namespace) -> None:
    """Run the validation-only hyperparameter search for every roster method and condition."""
    started = time.perf_counter()
    if args.split_index is not None:
        _tune_split(args, started)
        return
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


def _tune_split(args: argparse.Namespace, started: float) -> None:
    """Tune independently within one explicit patient split."""
    paths = split_paths(ensure_dirs(load_config(args.config)), args.split_index)
    if _is_excluded(paths):
        return
    paths, regime, loader = _tuning_inputs(args, paths)
    freeze = json.loads((paths["data"] / "manifest_freeze.json").read_text())
    selections = _split_selections(paths, regime, loader, freeze, args)
    write_json(paths["data"] / _output_name(args), selections)
    _write_tuning_cost(paths, started, getattr(args, "condition", None))


def _split_selections(
    paths: dict[str, Path],
    regime: Regime,
    loader: torch.utils.data.DataLoader,
    freeze: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, dict[str, Any]]:
    """Tune every requested condition and tail assignment for one split."""
    assignments = tuple(freeze.get("tail_assignments", {"native": []}))
    selections: dict[str, dict[str, Any]] = {
        assignment: {} for assignment in assignments
    }
    methods, seeds = roster_for_regime(regime.is_mil), _tuning_seeds(args.seed)
    for condition in _conditions(args):
        scoped = ("native",) if condition in {"natural", "balanced"} else assignments
        for assignment in scoped:
            selections[assignment][condition] = tune_condition(
                methods,
                loader,
                regime,
                seeds,
                paths["data"] / _manifest_name(condition, assignment),
            )
    return selections


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
        write_json(paths["data"] / _output_name(args), selections)
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
            dataset_cls = BagFeatureDataset if regime.is_mil else ImbalanceDataset
            result.append(
                TuningScope(
                    regime,
                    loader,
                    dataset_cls(
                        paths["data"] / _manifest_name(condition, assignment),
                        device=regime.device,
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
