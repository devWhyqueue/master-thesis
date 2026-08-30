from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Any

from imbalance_benchmark.common import sign_file, split_paths, write_json
from imbalance_benchmark.modeling.context import (
    CONDITIONS,
    roster_for_condition,
    scoped_assignments,
)
from imbalance_benchmark.modeling.workflows.tuning.aggregation.aggregate import (
    combined_cost,
)
from imbalance_benchmark.modeling.workflows.tuning.aggregation.candidate_registry import (
    load_round_grids,
    load_round_state,
    terminal_cost_payloads,
    tuning_locked,
)
from imbalance_benchmark.modeling.workflows.tuning.tuning_artifacts import (
    condition_is_reusable,
    expected_observations,
)
from imbalance_benchmark.modeling.workflows.tuning.tuning_reduction import (
    ReduceRound,
    reduce_phase,
    reduce_terminal_phase,
    terminal_active_grids,
)
from imbalance_benchmark.modeling.workflows.tuning.search_windows import expand_grid
from imbalance_benchmark.modeling.workflows.tuning.tuning_rounds import (
    new_configs_for_round,
)
from imbalance_benchmark.modeling.workflows.tuning.tuning_schedule import phase_methods


def _bundle_indices(
    bundle_index: int, size: int, observation_count: int, by_observation: bool
) -> list[int]:
    if not by_observation:
        first = bundle_index * size
        return list(range(first, first + size))
    candidate_group, observation_index = divmod(bundle_index, observation_count)
    first_candidate = candidate_group * size
    return [
        candidate_index * observation_count + observation_index
        for candidate_index in range(first_candidate, first_candidate + size)
    ]


def write_base_selection(
    root: Path, condition: str, selections: dict[str, Any]
) -> Path:
    """Merge signed base selections without dropping resolved methods."""
    path = root / "tuning_shards" / f"base_selections_{condition}.json"
    existing = json.loads(path.read_text()) if path.exists() else {}
    write_json(path, {**existing, **selections})
    sign_file(path)
    return path


def write_base_selections(
    base: dict[str, Path],
    freeze: dict[str, Any],
    fingerprint: list[str],
    is_mil: bool,
    conditions: tuple[str, ...],
    accepted: list[set[str]] | None = None,
) -> None:
    """Reduce and sign every incomplete base-method condition."""
    assignments = tuple(freeze.get("tail_assignments", {"native": []}))
    for condition in conditions:
        if not scoped_assignments(condition, freeze, assignments):
            continue  # not constructed for this dataset (plans/03,04)
        roster = roster_for_condition(is_mil, condition)
        if condition_is_reusable(base, condition, roster, assignments):
            continue
        selected, _ = reduce_phase(
            base["data"],
            condition,
            "base",
            phase_methods(is_mil, "base", condition),
            freeze["method_grids"],
            ReduceRound(fingerprint, accepted=accepted),
            expected_observations(condition, assignments, freeze),
        )
        write_base_selection(base["data"], condition, selected)


def write_final_selections(
    base: dict[str, Path],
    freeze: dict[str, Any],
    fingerprint: list[str],
    is_mil: bool,
    conditions: tuple[str, ...],
    accepted: list[set[str]] | None = None,
) -> None:
    """Write the unchanged signed selection interface and parallel search costs."""
    assignments = tuple(freeze.get("tail_assignments", {"native": []}))
    for condition in conditions:
        if not scoped_assignments(condition, freeze, assignments):
            continue  # not constructed for this dataset (plans/03,04)
        roster = roster_for_condition(is_mil, condition)
        if condition_is_reusable(base, condition, roster, assignments):
            continue
        _reduce_condition(
            base,
            freeze,
            fingerprint,
            phase_methods(is_mil, "base", condition),
            phase_methods(is_mil, "dependent", condition),
            assignments,
            condition,
            accepted,
        )


def _reduce_condition(
    base: dict[str, Path],
    freeze: dict[str, Any],
    fingerprint: list[str],
    base_methods: tuple[str, ...],
    dependent_methods: tuple[str, ...],
    assignments: tuple[str, ...],
    condition: str,
    accepted: list[set[str]] | None = None,
) -> None:
    """Reduce every required method's signed terminal adaptive state, not round 0.

    ``base_selections_*`` may only reflect a method's own resolving round,
    and dependent methods (e.g. ``crt``) never appear there at all - the
    terminal ``tuning_round_state`` is the only source of truth for both.
    """
    root = base["data"]
    methods = (*base_methods, *dependent_methods)
    if not tuning_locked(root, condition, methods):
        raise RuntimeError(f"Tuning is not locked for condition: {condition}")
    state = load_round_state(root, condition)
    terminal_grids = terminal_active_grids(state, methods, len(freeze["class_names"]))
    expected = expected_observations(condition, assignments, freeze)
    reduce_round = ReduceRound(fingerprint, accepted=accepted)
    base_selected, _ = reduce_terminal_phase(
        root, condition, "base", base_methods, terminal_grids, reduce_round, expected
    )
    dependent_selected, _ = reduce_terminal_phase(
        root,
        condition,
        "dependent",
        dependent_methods,
        terminal_grids,
        reduce_round,
        expected,
    )
    selected = {**base_selected, **dependent_selected}
    cost_payloads = [
        *terminal_cost_payloads(
            root, condition, "base", base_methods, fingerprint, expected, accepted
        ),
        *terminal_cost_payloads(
            root,
            condition,
            "dependent",
            dependent_methods,
            fingerprint,
            expected,
            accepted,
        ),
    ]
    scoped = scoped_assignments(condition, freeze, assignments)
    output = {assignment: {} for assignment in assignments}
    for assignment in scoped:
        output[assignment][condition] = selected
    _write_condition_outputs(base, condition, output, combined_cost(cost_payloads))


def _write_condition_outputs(
    base: dict[str, Path],
    condition: str,
    selections: dict[str, dict[str, Any]],
    cost: dict[str, Any],
) -> None:
    for index in range(3):
        paths = split_paths(base, index)
        selection_path = paths["data"] / f"tuning_selections_{condition}.json"
        write_json(selection_path, selections)
        sign_file(selection_path)
        write_json(paths["data"] / f"tuning_search_cost_{condition}.json", cost)


def reduce_tuning_shards(
    base: dict[str, Path],
    freeze: dict[str, Any],
    fingerprint: list[str],
    phase: str,
    condition: str | None = None,
    accepted: list[set[str]] | None = None,
) -> None:
    """Reduce complete base or dependent shards into signed selections.

    ``condition`` scopes a ``phase="final"`` reduce to one condition, since
    others may still be mid-search when this one converges. Reads only shard
    artifacts and the freeze record, so this never needs a loaded scope.
    """
    is_mil = freeze["runtime_config"].get("dataset", {}).get("regime") == "wsi"
    if phase == "base":
        write_base_selections(base, freeze, fingerprint, is_mil, CONDITIONS, accepted)
        return
    write_final_selections(
        base,
        freeze,
        fingerprint,
        is_mil,
        (condition,) if condition else CONDITIONS,
        accepted,
    )


def round_overridden_scopes(
    root: Path,
    condition: str,
    phase: str,
    scopes: list[tuple[dict[str, Path], Any, Any]],
    methods: tuple[str, ...],
) -> list[tuple[dict[str, Path], Any, Any]]:
    """Replace each split's frozen method_grids with this round's new-configs only.

    ``_fit_payload`` resolves a shard's config from ``regime.method_grids``,
    so training round>0's genuinely new candidates needs no change there,
    just a regime whose grids hold this round's values at the same indices
    ``resolve_round_shard_spec`` used to address them.
    """
    round_grids = load_round_grids(root, condition, phase)
    overrides = {
        method: new_configs_for_round(
            root, condition, method, expand_grid(**round_grids["windows"][method])
        )
        for method in methods
        if method in round_grids["windows"]
    }
    return [
        (
            paths,
            replace(regime, method_grids={**regime.method_grids, **overrides}),
            loader,
        )
        for paths, regime, loader in scopes
    ]
