from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Any

from imbalance_benchmark.common import sign_file, split_paths, write_json
from imbalance_benchmark.modeling.context import CONDITIONS, roster_for_regime
from imbalance_benchmark.modeling.workflows.tuning_aggregate import combined_cost
from imbalance_benchmark.modeling.workflows.tuning.candidate_registry import (
    load_round_grids,
)
from imbalance_benchmark.modeling.workflows.tuning.tuning_artifacts import (
    condition_is_reusable,
    expected_observations,
)
from imbalance_benchmark.modeling.workflows.tuning.tuning_reduction import (
    ReduceRound,
    reduce_phase,
)
from imbalance_benchmark.modeling.workflows.tuning.tuning_rounds import (
    expand_grid,
    new_configs_for_round,
)
from imbalance_benchmark.modeling.workflows.tuning.tuning_schedule import phase_methods


def _bundle_indices(
    bundle_index: int, size: int, observation_count: int, by_observation: bool
) -> list[int]:
    """Map one bundled SLURM task index to its flat candidate/observation indices."""
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
    """Merge and persist the signed base-method selection consumed by dependent shards.

    A later round only reduces its own still-unresolved subset, so this
    must merge rather than replace or a resolved method (e.g. ``ce``)
    would vanish once another method's round advances past it.
    """
    path = root / "tuning_shards" / f"base_selections_{condition}.json"
    existing = json.loads(path.read_text()) if path.exists() else {}
    write_json(path, {**existing, **selections})
    sign_file(path)
    return path


def write_base_selections(
    base: dict[str, Path],
    freeze: dict[str, Any],
    fingerprint: list[str],
    methods: tuple[str, ...],
    roster: tuple[str, ...],
    conditions: tuple[str, ...],
) -> None:
    """Reduce and sign every incomplete base-method condition."""
    assignments = tuple(freeze.get("tail_assignments", {"native": []}))
    for condition in conditions:
        if condition_is_reusable(base, condition, roster, assignments):
            continue
        selected, _ = reduce_phase(
            base["data"],
            condition,
            "base",
            methods,
            freeze["method_grids"],
            ReduceRound(fingerprint),
            expected_observations(condition, assignments, freeze),
        )
        write_base_selection(base["data"], condition, selected)


def write_final_selections(
    base: dict[str, Path],
    freeze: dict[str, Any],
    fingerprint: list[str],
    base_methods: tuple[str, ...],
    dependent_methods: tuple[str, ...],
    conditions: tuple[str, ...],
) -> None:
    """Write the unchanged signed selection interface and parallel search costs."""
    assignments = tuple(freeze.get("tail_assignments", {"native": []}))
    for condition in conditions:
        if condition_is_reusable(
            base, condition, (*base_methods, *dependent_methods), assignments
        ):
            continue
        _reduce_condition(
            base,
            freeze,
            fingerprint,
            base_methods,
            dependent_methods,
            assignments,
            condition,
        )


def _reduce_condition(
    base: dict[str, Path],
    freeze: dict[str, Any],
    fingerprint: list[str],
    base_methods: tuple[str, ...],
    dependent_methods: tuple[str, ...],
    assignments: tuple[str, ...],
    condition: str,
) -> None:
    reduce_round = ReduceRound(fingerprint)
    base_selected, base_payloads = reduce_phase(
        base["data"],
        condition,
        "base",
        base_methods,
        freeze["method_grids"],
        reduce_round,
        expected_observations(condition, assignments, freeze),
    )
    dependent, dependent_payloads = reduce_phase(
        base["data"],
        condition,
        "dependent",
        dependent_methods,
        freeze["method_grids"],
        reduce_round,
        expected_observations(condition, assignments, freeze),
    )
    selected = {**base_selected, **dependent}
    scoped = ("native",) if condition in {"natural", "balanced"} else assignments
    output = {assignment: {} for assignment in assignments}
    for assignment in scoped:
        output[assignment][condition] = selected
    _write_condition_outputs(
        base, condition, output, combined_cost([*base_payloads, *dependent_payloads])
    )


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
    raw_scopes: list[tuple[dict[str, Path], Any, Any]],
    freeze: dict[str, Any],
    fingerprint: list[str],
    phase: str,
    condition: str | None = None,
) -> None:
    """Reduce complete base or dependent shards into signed selections.

    ``condition`` scopes a ``phase="final"`` reduce to one condition, since
    the others may still be mid-search when this condition's converges.
    """
    is_mil = raw_scopes[0][1].is_mil
    base_methods = phase_methods(is_mil, "base")
    if phase == "base":
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
        (condition,) if condition else CONDITIONS,
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
