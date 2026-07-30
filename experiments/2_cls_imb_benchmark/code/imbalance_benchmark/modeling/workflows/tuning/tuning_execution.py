from __future__ import annotations

from pathlib import Path
from typing import Any

from dataclasses import replace

from imbalance_benchmark.modeling.context import CONDITIONS, roster_for_regime
from imbalance_benchmark.modeling.workflows.tuning.candidate_registry import (
    load_round_grids,
)
from imbalance_benchmark.modeling.workflows.tuning.tuning_reduction import (
    write_base_selections,
    write_final_selections,
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


def reduce_tuning_shards(
    base: dict[str, Path],
    raw_scopes: list[tuple[dict[str, Path], Any, Any]],
    freeze: dict[str, Any],
    fingerprint: list[str],
    phase: str,
) -> None:
    """Reduce complete base or dependent shards into signed selections."""
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
        CONDITIONS,
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
