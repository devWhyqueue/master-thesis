from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from imbalance_benchmark.commands.confirm import (
    _confirm_run_data,
    _is_excluded,
    _load_condition_selections,
    require_tuning_configs,
)
from imbalance_benchmark.common import (
    ensure_dirs,
    load_config,
    read_run_record,
    split_paths,
)
from imbalance_benchmark.manifest.freeze import accepted_freeze_hashes
from imbalance_benchmark.datasets.data import TrainDataset, load_training_dataset
from imbalance_benchmark.modeling.context import (
    MATCHED_BETA_METHOD,
    matched_beta_config,
    roster_for_condition,
    scoped_assignments,
)
from imbalance_benchmark.modeling.workflows.confirmation import (
    RunContext,
    confirm_ce_seed,
    confirm_crt_seed,
    confirm_method_seed,
    confirm_post_hoc_seed,
)
from imbalance_benchmark.modeling.workflows.confirmation_schedule import (
    ConfirmUnit,
    resolve_confirm_bundle,
)

__all__ = ["cmd_confirm_shard"]


def _fitted_methods(cond: str, method: str, is_mil: bool) -> tuple[str, ...]:
    """Methods one unit actually fits: post-hoc rides with ce where the roster has it."""
    if method == "ce" and "post_hoc_logit_adjustment" in roster_for_condition(
        is_mil, cond
    ):
        return ("ce", "post_hoc_logit_adjustment")
    return (method,)


def _required_methods(cond: str, method: str, is_mil: bool) -> tuple[str, ...]:
    """Methods whose signed tuning selection a unit needs."""
    if method == "crt":
        return ("crt", "ce")
    if method == MATCHED_BETA_METHOD:
        return ("independent_support_ce", "class_balanced_ce")
    return _fitted_methods(cond, method, is_mil)


def _unit_manifest_name(cond: str, assignment: str) -> str:
    return (
        f"manifest_{cond}.csv"
        if cond in {"natural", "balanced"}
        else f"manifest_{assignment}_{cond}.csv"
    )


def _confirm_unit_method(
    cond: str, method: str, seed_idx: int, best_configs: dict[str, Any], run: RunContext
) -> None:
    """Fit one roster method for one confirmation seed; post-hoc rides with ce."""
    train_ds: TrainDataset = load_training_dataset(
        run.paths["data"] / _unit_manifest_name(cond, run.assignment),
        run.is_mil,
        class_names=run.class_names,
    )
    configs = require_tuning_configs(
        run.paths["data"].parent.parent / "data",
        cond,
        best_configs,
        _required_methods(cond, method, run.is_mil),
    )
    if method == "ce":
        state, step = confirm_ce_seed(cond, configs["ce"], train_ds, run, seed_idx)
        if "post_hoc_logit_adjustment" in _fitted_methods(cond, method, run.is_mil):
            confirm_post_hoc_seed(
                cond,
                configs["post_hoc_logit_adjustment"],
                state,
                step,
                train_ds,
                run,
                seed_idx,
            )
    elif method == "crt":
        confirm_crt_seed(cond, configs["crt"], configs["ce"], train_ds, run, seed_idx)
    elif method == MATCHED_BETA_METHOD:
        confirm_method_seed(
            cond, method, matched_beta_config(configs), train_ds, run, seed_idx
        )
    else:
        confirm_method_seed(cond, method, configs[method], train_ds, run, seed_idx)


def _result_dir(
    paths: dict[str, Any], assignment: str, cond: str, method: str, seed_idx: int
) -> Path:
    return (
        paths["results"]
        / f"assignment={assignment}"
        / cond
        / method
        / f"seed={seed_idx}"
    )


def _seed_already_done(
    paths: dict[str, Any],
    assignment: str,
    cond: str,
    method: str,
    seed_idx: int,
    configs: dict[str, Any],
    is_mil: bool,
    accepted: set[str | None],
) -> bool:
    """Return whether complete records match the current effective configurations.

    A crash mid-write can leave a truncated ``run.json``; treat any read failure
    as not-done so a resumed task refits rather than trusting a corrupt record.
    A record stamped with a freeze hash outside ``accepted`` predates the
    current (or an amendment's superseded) freeze and must be refit too -
    otherwise a pre-refreeze run lingers forever, as ``ingest_all_runs``
    would refuse it at analyze time regardless.
    """
    for name in _fitted_methods(cond, method, is_mil):
        result_dir = _result_dir(paths, assignment, cond, name, seed_idx)
        try:
            record = read_run_record(result_dir)
        except (OSError, ValueError):
            return False
        if record is None or "test" not in record.get("splits", {}):
            return False
        if record.get("provenance", {}).get("freeze_content_sha256") not in accepted:
            return False
        selected = configs.get(name)
        if name == "post_hoc_logit_adjustment" and isinstance(selected, dict):
            selected = {"parameter": selected.get("parameter")}
        if name == "crt" and isinstance(selected, dict):
            selected = {**selected, "stage_one": configs.get("ce")}
        if name == MATCHED_BETA_METHOD:
            selected = matched_beta_config(configs)
        if record.get("tuning_params") != selected:
            return False
    return True


def _run_confirm_unit(
    unit: ConfirmUnit,
    best_configs: dict[str, Any],
    run_data: dict[str, Any],
    scoped: tuple[str, ...],
    accepted: set[str | None],
) -> None:
    """Fit one scheduled unit for every tail assignment scoped to its condition."""
    for assignment in scoped:
        selected_assignment = "native" if assignment == "unassigned" else assignment
        selected = best_configs.get(selected_assignment, {}).get(unit.condition, {})
        if _seed_already_done(
            run_data["paths"],
            assignment,
            unit.condition,
            unit.method,
            unit.seed_index,
            selected,
            run_data["is_mil"],
            accepted,
        ):
            continue
        run = RunContext(**run_data, assignment=assignment)
        _confirm_unit_method(
            unit.condition, unit.method, unit.seed_index, selected, run
        )


def _group_bundle_by_split(units: list[ConfirmUnit]) -> dict[int, list[ConfirmUnit]]:
    grouped: dict[int, list[ConfirmUnit]] = {}
    for unit in units:
        grouped.setdefault(unit.split_index, []).append(unit)
    return grouped


def _run_split_bundle(paths: dict[str, Any], split_units: list[ConfirmUnit]) -> None:
    """Fit every scheduled unit for one split, skipping conditions this dataset never built."""
    run_data, freeze = _confirm_run_data(paths)
    assignments = tuple(freeze.get("tail_assignments", {"native": []}))
    accepted = accepted_freeze_hashes(freeze)
    selections: dict[str, dict[str, Any]] = {}
    for unit in split_units:
        scoped = scoped_assignments(unit.condition, freeze, assignments, "unassigned")
        if not scoped:
            continue  # not constructed for this dataset (plans/03,04)
        if unit.condition not in selections:
            selections[unit.condition] = _load_condition_selections(
                paths, unit.condition
            )
        _run_confirm_unit(unit, selections[unit.condition], run_data, scoped, accepted)


def cmd_confirm_shard(args: argparse.Namespace) -> None:
    """Run one resumable bundle of confirmation units for one partition group.

    A bundle can span multiple conditions (the controlled group has three), so
    selections are loaded per condition from the ``tune-final-reduce`` output
    and cached only within the split currently being processed.
    """
    config = load_config(args.config)
    base_paths = ensure_dirs(config)
    is_mil = config.get("dataset", {}).get("regime", "patch") == "wsi"
    bundle = resolve_confirm_bundle(
        args.shard_index, args.group, is_mil, args.shards_per_task
    )
    for split_index, split_units in _group_bundle_by_split(bundle).items():
        paths = split_paths(base_paths, split_index)
        if _is_excluded(paths):
            continue
        _run_split_bundle(paths, split_units)
