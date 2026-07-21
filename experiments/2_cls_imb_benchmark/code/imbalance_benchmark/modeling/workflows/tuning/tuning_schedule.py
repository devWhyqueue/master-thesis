from __future__ import annotations

from dataclasses import replace
from typing import Any

from imbalance_benchmark.modeling.context import roster_for_regime
from imbalance_benchmark.modeling.workflows.tuning.tuning_artifacts import ShardSpec

MAX_CANDIDATES = 16
DEPENDENT_METHODS = ("post_hoc_logit_adjustment", "crt")


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
    conditions = (
        ("natural",) if group == "natural" else ("balanced", "moderate", "severe")
    )
    per_condition = candidate_array_size(methods)
    condition_index, remainder = divmod(index, per_condition)
    if condition_index >= len(conditions):
        return None
    method, candidate_index = _candidate_slot(methods, remainder)
    limit = 1 if method == "post_hoc_logit_adjustment" else len(grids.get(method, []))
    if candidate_index >= limit:
        return None
    return ShardSpec(conditions[condition_index], method, candidate_index, phase)


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


def phase_methods(is_mil: bool, phase: str) -> tuple[str, ...]:
    """Split the frozen roster around the CE-dependent methods."""
    roster = roster_for_regime(is_mil)
    return tuple(
        method
        for method in roster
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
    spec = resolve_shard_spec(index, phase, group, phase_methods(is_mil, phase), grids)
    return replace(spec, observation_index=observation_index) if spec else None


def array_coordinates(
    index: int, explicit_observation: int | None, observations_per_candidate: int
) -> tuple[int, int | None]:
    """Decode a candidate array task, optionally crossed with observations."""
    if observations_per_candidate < 1:
        raise ValueError("observations-per-candidate must be positive")
    if explicit_observation is not None and observations_per_candidate != 1:
        raise ValueError("Use either observation-index or observations-per-candidate")
    if observations_per_candidate == 1:
        return index, explicit_observation
    return divmod(index, observations_per_candidate)
