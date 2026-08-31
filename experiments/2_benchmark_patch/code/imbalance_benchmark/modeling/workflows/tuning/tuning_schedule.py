from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
from typing import Any

import torch

from imbalance_benchmark.common import split_paths, verify_signed_file
from imbalance_benchmark.datasets.data import load_training_dataset
from imbalance_benchmark.modeling.context import group_conditions, roster_for_condition
from imbalance_benchmark.modeling.workflows.tuning.aggregation.aggregate import (
    TuningScope,
)
from imbalance_benchmark.modeling.workflows.tuning.aggregation.candidate_registry import (
    load_round_grids,
)
from imbalance_benchmark.modeling.workflows.tuning.aggregation.tuning_budget import (
    tuning_example_budget,
)
from imbalance_benchmark.modeling.workflows.tuning.tuning_artifacts import (
    ShardSpec,
    _fingerprint_matches,
)
from imbalance_benchmark.modeling.workflows.tuning.search_windows import expand_grid
from imbalance_benchmark.modeling.workflows.tuning.tuning_rounds import (
    new_configs_for_round,
)

MAX_CANDIDATES = 16
DEPENDENT_METHODS = ("post_hoc_logit_adjustment", "crt")


def combined_scopes(
    raw_scopes: list[tuple[dict[str, Path], Any, torch.utils.data.DataLoader]],
    condition: str,
    assignments: tuple[str, ...],
    cost_records: list[dict[str, int]] | None = None,
) -> list[TuningScope]:
    """Build canonical assignment-then-split tuning observations."""
    records = cost_records if cost_records is not None else []
    width = len(raw_scopes)
    return [
        TuningScope(
            regime,
            loader,
            load_training_dataset(
                paths["data"] / _manifest_name(condition, assignment),
                regime.is_mil,
                class_names=regime.locked_class_names,
            ),
            records,
            tuning_example_budget(regime, condition),
            assignment,
            split_index,
            assignment_index * width + split_index,
        )
        for assignment_index, assignment in enumerate(assignments)
        for split_index, (paths, regime, loader) in enumerate(raw_scopes)
    ]


def _manifest_name(condition: str, assignment: str) -> str:
    return (
        f"manifest_{condition}.csv"
        if condition in {"natural", "balanced"}
        else f"manifest_{assignment}_{condition}.csv"
    )


def bundled_array_size(shard_count: int, shards_per_task: int) -> int:
    """Return the number of allocations needed for fixed-size shard bundles."""
    if shards_per_task < 1:
        raise ValueError("shards per task must be positive")
    return math.ceil(shard_count / shards_per_task)


def bundled_observation_array_size(
    candidate_count: int, observation_count: int, shards_per_task: int
) -> int:
    """Return allocations for candidate bundles crossed with observations."""
    return bundled_array_size(candidate_count, shards_per_task) * observation_count


def resolve_shard_spec(
    index: int,
    phase: str,
    group: str,
    methods: tuple[str, ...],
    grids: dict[str, list[dict[str, Any]]],
) -> ShardSpec | None:
    """Map a stable SLURM index to one valid frozen candidate."""
    if index < 0:
        return None
    conditions = group_conditions(group)
    per_condition = candidate_array_size(methods)
    condition_index, remainder = divmod(index, per_condition)
    if condition_index >= len(conditions):
        return None
    method, candidate_index = _candidate_slot(methods, remainder)
    limit = 1 if method == "post_hoc_logit_adjustment" else len(grids.get(method, []))
    if candidate_index >= limit:
        return None
    return ShardSpec(conditions[condition_index], method, candidate_index, phase)


def _new_configs_by_method(
    root: Path, condition: str, round_grids: dict[str, Any], methods: tuple[str, ...]
) -> dict[str, list[dict[str, Any]]]:
    """This round's genuinely new per-method configs, keyed by method."""
    return {
        method: new_configs_for_round(
            root, condition, method, expand_grid(**round_grids["windows"][method])
        )
        for method in methods
        if method in round_grids["windows"]
    }


def resolve_round_shard_spec(
    root: Path, condition: str, index: int, phase: str, methods: tuple[str, ...]
) -> ShardSpec | None:
    """Resolve one round>0 array index from this round's signed active
    windows: only genuinely new (not-yet-trained) configs are addressed,
    reusing round 0's exact fixed-slot addressing over the smaller grid."""
    round_grids = load_round_grids(root, condition, phase)
    grids = _new_configs_by_method(root, condition, round_grids, methods)
    spec = resolve_shard_spec(index, phase, "natural", tuple(grids), grids)
    if spec is None:
        return None
    return replace(spec, condition=condition, round=round_grids["round"])


def candidate_array_size(methods: tuple[str, ...]) -> int:
    """Return the compact fixed-slot array size for a method set."""
    return sum(_candidate_slots(method) for method in methods)


def _candidate_slots(method: str) -> int:
    if method == "post_hoc_logit_adjustment":
        return 1
    if method in {"ce", "crt"}:
        return 4
    return MAX_CANDIDATES


def _candidate_slot(methods: tuple[str, ...], index: int) -> tuple[str, int]:
    for method in methods:
        slots = _candidate_slots(method)
        if index < slots:
            return method, index
        index -= slots
    raise IndexError("Candidate shard index lies outside the method roster")


def phase_methods(is_mil: bool, phase: str, condition: str) -> tuple[str, ...]:
    """Split the frozen roster into self-contained (``base``) methods and
    those that inherit CE's tuned config (``dependent``)."""
    return tuple(
        method
        for method in roster_for_condition(is_mil, condition)
        if (method not in DEPENDENT_METHODS) == (phase == "base")
    )


def requested_shard(
    index: int,
    phase: str,
    group: str,
    is_mil: bool,
    grids: dict[str, list[dict[str, Any]]],
    observation_index: int | None,
) -> ShardSpec | None:
    """Resolve a CLI array index and optional observation fallback index."""
    # Every condition in a group shares one roster, so its first fixes the methods.
    methods = phase_methods(is_mil, phase, group_conditions(group)[0])
    spec = resolve_shard_spec(index, phase, group, methods, grids)
    return replace(spec, observation_index=observation_index) if spec else None


def _load_signed_condition_artifacts(
    base: dict[str, Path], condition: str
) -> tuple[list[Any], list[Any]] | None:
    """Load every split's signed selection and cost file, or None if any is bad."""
    selections, costs = [], []
    for index in range(3):
        data = split_paths(base, index)["data"]
        selection = data / f"tuning_selections_{condition}.json"
        cost = data / f"tuning_search_cost_{condition}.json"
        try:
            verify_signed_file(selection)
            selections.append(json.loads(selection.read_text()))
            costs.append(json.loads(cost.read_text()))
        except (FileNotFoundError, json.JSONDecodeError, RuntimeError):
            return None
    return selections, costs


def condition_is_reusable(
    base: dict[str, Path],
    condition: str,
    methods: tuple[str, ...],
    assignments: tuple[str, ...],
    fingerprint: list[str],
    accepted: list[set[str]] | None = None,
) -> bool:
    """Accept a completed serial condition only when every split matches and
    its recorded fingerprint still chains (via ``accepted``) to the current
    freeze - otherwise a stale pre-amendment selection is reused forever."""
    loaded = _load_signed_condition_artifacts(base, condition)
    if loaded is None:
        return False
    selections, costs = loaded
    if selections[1:] != selections[:1] * 2 or costs[1:] != costs[:1] * 2:
        return False
    required_cost = {"wall_clock_seconds", "accelerator_hours", "processed_examples"}
    if not required_cost.issubset(costs[0]):
        return False
    if not _fingerprint_matches(costs[0].get("fingerprint"), fingerprint, accepted):
        return False
    scoped = ("native",) if condition in {"natural", "balanced"} else assignments
    return all(
        all(
            isinstance(selections[0].get(name, {}).get(condition, {}).get(method), dict)
            for method in methods
        )
        for name in scoped
    )


def array_coordinates(
    index: int,
    explicit_observation: int | None,
    observations_per_candidate: int,
    candidate_offset: int = 0,
) -> tuple[int, int | None]:
    """Decode a candidate array task, optionally crossed with observations."""
    if observations_per_candidate < 1:
        raise ValueError("observations-per-candidate must be positive")
    if candidate_offset < 0:
        raise ValueError("candidate offset must be non-negative")
    if explicit_observation is not None and observations_per_candidate != 1:
        raise ValueError("Use either observation-index or observations-per-candidate")
    if observations_per_candidate == 1:
        return index + candidate_offset, explicit_observation
    candidate_index, observation_index = divmod(index, observations_per_candidate)
    return candidate_index + candidate_offset, observation_index
