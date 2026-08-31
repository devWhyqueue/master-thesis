from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import replace
import multiprocessing
import os
from pathlib import Path
import time
from typing import Any

import torch

from imbalance_benchmark.commands.tuning import (
    _frozen_shard_context,
    _is_excluded,
    _tuning_seeds,
)
from imbalance_benchmark.modeling.context import Regime, roster_for_condition
from imbalance_benchmark.modeling.workflows.tuning.aggregation.aggregate import (
    TuningScope,
)
from imbalance_benchmark.modeling.workflows.tuning.tuning_execution import (
    round_overridden_scopes,
)
from imbalance_benchmark.modeling.workflows.tuning.tuning_artifacts import selected_ce
from imbalance_benchmark.modeling.workflows.tuning.tuning_reduction import ReduceRound
from imbalance_benchmark.modeling.workflows.tuning.tuning_shards import (
    ShardSpec,
    run_candidate_shard,
)
from imbalance_benchmark.modeling.workflows.tuning.tuning_schedule import (
    array_coordinates,
    combined_scopes,
    condition_is_reusable,
    phase_methods,
    resolve_round_shard_spec,
)

__all__ = ["run_round_shards"]

# A fresh spawned interpreter per child avoids re-initializing CUDA in a forked
# process, and gives each fit its own process-local feature bank (cache.py's
# ``_BANK`` is a module global). Forcing cuda placement below removes the
# ``torch.cuda.mem_get_info()`` race between concurrently-starting children;
# staggering their starts additionally spreads out CUDA context init itself.
_PARALLEL_FIT_ENV = {"IMB_FEATURE_BANK_DEVICE": "cuda"}
_CHILD_STAGGER_SECONDS = 3.0


def _chunk(specs: list[ShardSpec], workers: int) -> list[list[ShardSpec]]:
    """Split specs round-robin across up to ``workers`` non-empty chunks."""
    chunks: list[list[ShardSpec]] = [[] for _ in range(workers)]
    for position, spec in enumerate(specs):
        chunks[position % workers].append(spec)
    return [batch for batch in chunks if batch]


def _run_and_join(processes: list[Any]) -> None:
    for index, process in enumerate(processes):
        process.start()
        if index < len(processes) - 1:
            time.sleep(_CHILD_STAGGER_SECONDS)
    for process in processes:
        process.join()
    failed = [str(index) for index, process in enumerate(processes) if process.exitcode]
    if failed:
        raise RuntimeError(f"Tuning shard workers failed: {', '.join(failed)}")


def _run_chunk(run_one: Callable[[ShardSpec], None], specs: list[ShardSpec]) -> None:
    os.environ.update(_PARALLEL_FIT_ENV)
    for spec in specs:
        run_one(spec)


def _run_packed(
    run_one: Callable[[ShardSpec], None], specs: list[ShardSpec], parallel_fits: int
) -> None:
    """Run every spec via ``run_one``, packing up to ``parallel_fits`` fits per GPU.

    Below 2 workers this runs sequentially in-process (bit-identical to no
    packing at all). Above that, each worker is a fresh spawned interpreter
    holding its own chunk of specs -- CUDA cannot be reinitialized in a
    forked process, and a fresh process also gives each fit an independent,
    race-free view of ``cache._target_bank_device``'s VRAM check. ``run_one``
    is expected to already be bound to its fixed (args, base, freeze,
    fingerprint, accepted) context, e.g. via ``functools.partial``.
    """
    workers = min(max(1, parallel_fits), len(specs))
    if workers <= 1:
        for spec in specs:
            run_one(spec)
        return
    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(target=_run_chunk, args=(run_one, batch))
        for batch in _chunk(specs, workers)
    ]
    _run_and_join(processes)


def run_round_shards(args: argparse.Namespace, indices: list[int]) -> None:
    """Run round>0 candidate indices: only genuinely new configs are trained."""
    if args.condition is None:
        raise ValueError("--condition is required for a round>0 shard")
    base, scopes, freeze, fingerprint, accepted = _frozen_shard_context(args)
    if any(_is_excluded(paths) for paths, _, _ in scopes):
        return
    methods = phase_methods(scopes[0][1].is_mil, args.phase, args.condition)
    overridden = round_overridden_scopes(
        base["data"], args.condition, args.phase, scopes, methods
    )

    def _spec_for(index: int) -> ShardSpec | None:
        candidate, observation = array_coordinates(
            index, args.observation_index, args.observations_per_candidate
        )
        spec = resolve_round_shard_spec(
            base["data"], args.condition, candidate, args.phase, methods
        )
        return replace(spec, observation_index=observation) if spec else None

    built: dict[tuple[str, tuple[str, ...]], list[TuningScope]] = {}
    for index in indices:
        spec = _spec_for(index)
        if spec is not None:
            _run_shard(base, overridden, freeze, fingerprint, accepted, built, spec)


def _run_shard(
    base: dict[str, Path],
    raw_scopes: list[tuple[dict[str, Path], Regime, torch.utils.data.DataLoader]],
    freeze: dict[str, Any],
    fingerprint: list[str],
    accepted: list[set[str]],
    built: dict[tuple[str, tuple[str, ...]], list[TuningScope]],
    spec: ShardSpec,
) -> None:
    """Run one shard, reusing ``built``'s cached scopes per (condition, scoped) key."""
    assignments = tuple(freeze.get("tail_assignments", {"native": []}))
    if condition_is_reusable(
        base,
        spec.condition,
        roster_for_condition(raw_scopes[0][1].is_mil, spec.condition),
        assignments,
        fingerprint,
        accepted,
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
        ReduceRound(fingerprint, accepted=accepted),
        base["data"],
        selected_ce(base["data"], spec.condition)
        if spec.phase == "dependent"
        else None,
    )
