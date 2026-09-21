from __future__ import annotations

import argparse
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

from imbalance_benchmark.common import split_paths
from imbalance_benchmark.commands.tuning import (
    _frozen_shard_context,
    _is_excluded,
    _tuning_seeds,
    bank_bytes_for,
    load_shard_scope,
)
from imbalance_benchmark.commands.tuning.shard_workers import (
    _run_packed,
    run_round_shards,
)
from imbalance_benchmark.datasets.features.cache import reset_feature_bank
from imbalance_benchmark.modeling.context import roster_for_condition
from imbalance_benchmark.modeling.workflows.tuning.tuning_execution import (
    _bundle_indices,
)
from imbalance_benchmark.modeling.workflows.tuning.tuning_artifacts import (
    expected_observations,
    selected_ce,
)
from imbalance_benchmark.modeling.workflows.tuning.tuning_reduction import ReduceRound
from imbalance_benchmark.modeling.workflows.tuning.tuning_shards import (
    ShardSpec,
    run_candidate_shard,
)
from imbalance_benchmark.modeling.workflows.tuning.tuning_schedule import (
    array_coordinates,
    condition_is_reusable,
    requested_shard,
)

__all__ = ["cmd_tune_shard"]

# Fixed per-fit overhead beyond the bank: activations, optimizer state, and
# -- dominant -- a handful of methods' own extra tensors (e.g. a metric-
# learning method's full-batch pairwise-distance eval). Measured 2026-09-01:
# tcga_ut/patch peaks up to 19.29 GiB/fit vs an analytic ~1.46 GiB controlled
# bank -- an ~17.8 GiB method-driven gap, not a bank-size one. Rounded up.
_MEASURED_TRANSIENT_BYTES = 18 * 1024**3


def _max_bank_bytes(
    base: dict[str, Path], freeze: dict[str, Any], specs: list[ShardSpec]
) -> int:
    """Worst per-fit VRAM footprint in this bundle, 0 if unknown (disables the cap)."""
    conditions = {spec.condition for spec in specs if isinstance(spec, ShardSpec)}
    if not conditions:
        return 0
    assignments = tuple(freeze.get("tail_assignments", {"native": []}))
    dtype = (
        freeze.get("runtime_config", {}).get("feature_extraction", {}).get("dtype")
        or "float16"
    )
    try:
        return max(
            bank_bytes_for(base, condition, assignment, dtype)
            + _MEASURED_TRANSIENT_BYTES
            for condition in conditions
            for assignment in (
                ("native",) if condition in {"natural", "balanced"} else assignments
            )
        )
    except (KeyError, OSError):
        return 0  # unreadable manifest -- a packing optimization, not correctness


# p95 per-fit seconds measured 2026-09-01 across tcga_ut/patch's heaviest
# (18-observation) controlled conditions; seeds each DeadlineGuard before it
# has timed a real item, which then widens the margin to what it observes.
_MEASURED_PER_FIT_SECONDS = 220.0


def _max_item_seconds(freeze: dict[str, Any], specs: list[ShardSpec]) -> float:
    """Worst-case wall time the next item can take, 0 if unknown (no-op guard).

    A round-0 non-bundled spec covers a whole candidate (its full observation
    count); a bundled-by-observation spec covers exactly one fit.
    """
    real = [spec for spec in specs if isinstance(spec, ShardSpec)]
    if not real:
        return 0.0
    assignments = tuple(freeze.get("tail_assignments", {"native": []}))
    try:
        return _MEASURED_PER_FIT_SECONDS * max(
            1
            if spec.observation_index is not None
            else expected_observations(spec.condition, assignments, freeze)
            for spec in real
        )
    except KeyError:
        return 0.0


def _execute_shards(
    base: dict[str, Path],
    args: argparse.Namespace,
    freeze: dict[str, Any],
    fingerprint: list[str],
    accepted: list[set[str]],
    indices: list[int],
    spec_for: Callable[[int], ShardSpec | None],
    parallel_fits: int = 1,
) -> None:
    """Run every resolvable index, packing up to ``parallel_fits`` fits per GPU.

    ``indices`` is resolved to concrete specs here, once, by ``spec_for`` --
    the sole authority for the index-to-work-item mapping. Any packed
    children only ever receive already-resolved specs.
    """
    specs = [
        spec for spec in (spec_for(index) for index in indices) if spec is not None
    ]
    run_one = partial(_run_scope_local_shard, args, base, freeze, fingerprint, accepted)
    _run_packed(
        run_one,
        specs,
        parallel_fits,
        _max_bank_bytes(base, freeze, specs),
        _max_item_seconds(freeze, specs),
    )


def _run_shards(args: argparse.Namespace, indices: list[int]) -> None:
    """Run candidate indices with one loaded frozen MIL context, packed per GPU."""
    if args.group is None:
        raise ValueError("--group is required for a round-0 shard")
    base, _, freeze, fingerprint, accepted = _frozen_shard_context(args, False)
    if any(_is_excluded(paths) for paths in _split_paths(base)):
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
            freeze["runtime_config"].get("dataset", {}).get("regime") == "wsi",
            freeze["method_grids"],
            observation,
        )

    _execute_shards(
        base,
        args,
        freeze,
        fingerprint,
        accepted,
        indices,
        _spec_for,
        getattr(args, "parallel_fits", 1),
    )


def _split_paths(base: dict[str, Path]) -> list[dict[str, Path]]:
    """Return all frozen split directories without constructing feature datasets."""
    return [split_paths(base, index) for index in range(3)]


def _run_scope_local_shard(
    args: argparse.Namespace,
    base: dict[str, Path],
    freeze: dict[str, Any],
    fingerprint: list[str],
    accepted: list[set[str]],
    spec: ShardSpec,
) -> None:
    """Train one base shard with one validation-plus-training feature bank at a time."""
    is_mil = freeze["runtime_config"].get("dataset", {}).get("regime") == "wsi"
    assignments = tuple(freeze.get("tail_assignments", {"native": []}))
    if condition_is_reusable(
        base,
        spec.condition,
        roster_for_condition(is_mil, spec.condition),
        assignments,
        fingerprint,
        accepted,
    ):
        return
    scoped = ("native",) if spec.condition in {"natural", "balanced"} else assignments
    descriptors = [
        {"scope_index": index, "assignment": assignment, "split_index": split_index}
        for index, (assignment, split_index) in enumerate(
            (assignment, split_index)
            for assignment in scoped
            for split_index in range(3)
        )
    ]
    selected = descriptors
    if spec.observation_index is not None:
        selected = [descriptors[spec.observation_index // len(_tuning_seeds(freeze))]]

    def _scopes() -> Any:
        for descriptor in selected:
            reset_feature_bank()
            try:
                yield load_shard_scope(
                    args,
                    base,
                    spec.condition,
                    str(descriptor["assignment"]),
                    int(descriptor["split_index"]),
                    int(descriptor["scope_index"]),
                    [],
                )
            finally:
                reset_feature_bank()

    run_candidate_shard(
        spec,
        [],
        _tuning_seeds(freeze),
        ReduceRound(fingerprint, accepted=accepted),
        base["data"],
        selected_ce(base["data"], spec.condition)
        if spec.phase == "dependent"
        else None,
        scope_stream=(_scopes, len(descriptors), descriptors),
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
        run_round_shards(args, indices)
    else:
        _run_shards(args, indices)
