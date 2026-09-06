"""Resumable execution of one SLURM array task's bundle of fit work items."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from imbalance_benchmark.common import ensure_dirs, read_run_record, split_paths
from imbalance_benchmark.datasets.data import TrainDataset, load_training_dataset
from imbalance_benchmark.hydra.guards import DeadlineGuard
from imbalance_benchmark.manifest.freeze import accepted_freeze_hashes
from imbalance_benchmark.modeling.workflows.confirmation import (
    RunContext,
    confirm_ce_seed,
    confirm_method_seed,
)

from diversity.fit.context import fit_run_data, load_selection, method_config
from diversity.fit.units import FitUnit, resolve_fit_bundle
from diversity.manifests import exp2_split_paths

__all__ = ["run_fit_shard"]

# Single-orchestration fit cost measured for exp-2's own confirm-shard
# (commands/confirm/shard.py); exp-3 fits one method per unit (no per-
# assignment loop), so no multiplier is applied here.
_MEASURED_UNIT_SECONDS = 220.0


def _result_dir(
    paths: dict[str, Path], level: str, allocation: str, method: str, seed: int
) -> Path:
    return (
        paths["results"] / f"assignment={level}" / allocation / method / f"seed={seed}"
    )


def _seed_already_done(
    paths: dict[str, Path],
    level: str,
    allocation: str,
    method: str,
    seed_index: int,
    selected: dict[str, Any],
    accepted: set[str | None],
) -> bool:
    """Whether a complete, current-freeze, current-selection record already exists.

    Mirrors exp-2's ``commands/confirm/shard.py::_seed_already_done`` resumability
    contract so a resumed SLURM task refits only what is missing or stale.
    """
    result_dir = _result_dir(paths, level, allocation, method, seed_index)
    try:
        record = read_run_record(result_dir)
    except (OSError, ValueError):
        return False
    if record is None or "test" not in record.get("splits", {}):
        return False
    if record.get("provenance", {}).get("freeze_content_sha256") not in accepted:
        return False
    return record.get("tuning_params") == selected


def _run_fit_unit(
    exp2_paths: dict[str, Path],
    exp3_paths: dict[str, Path],
    freeze: dict[str, Any],
    run_data: dict[str, Any],
    unit: FitUnit,
) -> bool:
    """Fit one work item unless a matching record already exists; return whether it ran."""
    selections = load_selection(exp2_paths["data"], unit.allocation)
    selected = method_config(selections, unit.allocation, unit.method)
    accepted = accepted_freeze_hashes(freeze)
    if _seed_already_done(
        exp3_paths,
        unit.level,
        unit.allocation,
        unit.method,
        unit.seed_index,
        selected,
        accepted,
    ):
        return False
    manifest_path = exp3_paths["data"] / f"manifest_{unit.allocation}_{unit.level}.csv"
    train_ds: TrainDataset = load_training_dataset(
        manifest_path, False, class_names=run_data["class_names"]
    )
    run = RunContext(**run_data, assignment=unit.level)
    if unit.method == "ce":
        confirm_ce_seed(unit.allocation, selected, train_ds, run, unit.seed_index)
    else:
        confirm_method_seed(
            unit.allocation, unit.method, selected, train_ds, run, unit.seed_index
        )
    return True


def run_fit_shard(
    config: dict[str, Any], group: str, shard_index: int, shards_per_task: int
) -> None:
    """Fit one resumable bundle of work items for one SLURM array task."""
    units = resolve_fit_bundle(group, shard_index, shards_per_task)
    guard = DeadlineGuard(_MEASURED_UNIT_SECONDS)
    cache: dict[int, tuple[dict[str, Any], dict[str, Any]]] = {}
    for unit in units:
        if guard.should_stop():
            break
        if unit.split_index not in cache:
            exp3_paths = split_paths(ensure_dirs(config), unit.split_index)
            cache[unit.split_index] = fit_run_data(exp3_paths)
        run_data, freeze = cache[unit.split_index]
        exp2_paths = exp2_split_paths(config, unit.split_index)
        guard.start_item()
        _run_fit_unit(exp2_paths, run_data["paths"], freeze, run_data, unit)
        guard.finish_item()
