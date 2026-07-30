from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from imbalance_benchmark.commands.tuning import (
    _frozen_shard_context,
    _is_excluded,
    _tuning_seeds,
)
from imbalance_benchmark.modeling.context import Regime, roster_for_regime
from imbalance_benchmark.modeling.workflows.tuning_aggregate import TuningScope
from imbalance_benchmark.modeling.workflows.tuning.tuning_execution import (
    _bundle_indices,
    round_overridden_scopes,
)
from imbalance_benchmark.modeling.workflows.tuning.tuning_artifacts import (
    condition_is_reusable,
    selected_ce,
)
from imbalance_benchmark.modeling.workflows.tuning.tuning_shards import (
    ShardSpec,
    combined_scopes,
    run_candidate_shard,
)
from imbalance_benchmark.modeling.workflows.tuning.tuning_schedule import (
    array_coordinates,
    phase_methods,
    requested_shard,
    resolve_round_shard_spec,
)

__all__ = ["cmd_tune_shard"]


def _execute_shards(
    base: dict[str, Path],
    scopes: list[tuple[dict[str, Path], Regime, torch.utils.data.DataLoader]],
    freeze: dict[str, Any],
    fingerprint: list[str],
    indices: list[int],
    spec_for: Callable[[int], ShardSpec | None],
) -> None:
    """Run every resolvable index, reusing built scopes per (condition, scoped) key."""
    built: dict[tuple[str, tuple[str, ...]], list[TuningScope]] = {}
    for index in indices:
        spec = spec_for(index)
        if spec is not None:
            _run_shard(base, scopes, freeze, fingerprint, built, spec)


def _run_shards(args: argparse.Namespace, indices: list[int]) -> None:
    """Run candidate indices sequentially with one loaded frozen MIL context."""
    if args.group is None:
        raise ValueError("--group is required for a round-0 shard")
    base, scopes, freeze, fingerprint = _frozen_shard_context(args)
    if any(_is_excluded(paths) for paths, _, _ in scopes):
        return

    def _spec_for(index: int) -> ShardSpec | None:
        shard, observation = array_coordinates(
            index,
            args.observation_index,
            args.observations_per_candidate,
            args.shard_offset,
        )
        return requested_shard(
            shard,
            args.phase,
            args.group,
            scopes[0][1].is_mil,
            freeze["method_grids"],
            observation,
        )

    _execute_shards(base, scopes, freeze, fingerprint, indices, _spec_for)


def _run_round_shards(args: argparse.Namespace, indices: list[int]) -> None:
    """Run round>0 candidate indices: only genuinely new configs are trained."""
    if args.condition is None:
        raise ValueError("--condition is required for a round>0 shard")
    base, scopes, freeze, fingerprint = _frozen_shard_context(args)
    if any(_is_excluded(paths) for paths, _, _ in scopes):
        return
    methods = phase_methods(scopes[0][1].is_mil, args.phase)
    overridden = round_overridden_scopes(base["data"], args.condition, scopes, methods)

    def _spec_for(index: int) -> ShardSpec | None:
        return resolve_round_shard_spec(
            base["data"], args.condition, index, args.phase, methods
        )

    _execute_shards(base, overridden, freeze, fingerprint, indices, _spec_for)


def _run_shard(
    base: dict[str, Path],
    raw_scopes: list[tuple[dict[str, Path], Regime, torch.utils.data.DataLoader]],
    freeze: dict[str, Any],
    fingerprint: list[str],
    built: dict[tuple[str, tuple[str, ...]], list[TuningScope]],
    spec: ShardSpec,
) -> None:
    """Run one shard, reusing ``built``'s cached scopes per (condition, scoped) key."""
    assignments = tuple(freeze.get("tail_assignments", {"native": []}))
    if condition_is_reusable(
        base, spec.condition, roster_for_regime(raw_scopes[0][1].is_mil), assignments
    ):
        return
    scoped = ("native",) if spec.condition in {"natural", "balanced"} else assignments
    key = (spec.condition, scoped)
    if key not in built:
        built[key] = combined_scopes(raw_scopes, spec.condition, scoped, [])
    fresh_cost_records: list[dict[str, int]] = []
    run_candidate_shard(
        spec,
        [replace(scope, cost_records=fresh_cost_records) for scope in built[key]],
        _tuning_seeds(freeze),
        fingerprint,
        base["data"],
        selected_ce(base["data"], spec.condition)
        if spec.phase == "dependent"
        else None,
    )


def cmd_tune_shard(args: argparse.Namespace) -> None:
    """Run one resumable frozen-candidate shard."""
    if args.shards_per_task < 1:
        raise ValueError("shards-per-task must be positive")
    indices = _bundle_indices(
        args.shard_index,
        args.shards_per_task,
        args.observations_per_candidate,
        args.bundle_by_observation,
    )
    if getattr(args, "round", 0):
        _run_round_shards(args, indices)
    else:
        _run_shards(args, indices)
