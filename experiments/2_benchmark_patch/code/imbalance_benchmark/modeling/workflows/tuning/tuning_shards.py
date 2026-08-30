from __future__ import annotations

from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import time
from collections.abc import Callable, Iterator
from typing import Any

import torch

from imbalance_benchmark.modeling.workflows.tuning.aggregation.aggregate import (
    TuningScope,
    _evaluate,
    _select_post_hoc,
)
from imbalance_benchmark.modeling.workflows.tuning.tuning_artifacts import (
    ShardSpec,
    shard_path,
    validate_shard_payload,
    write_atomic as _write_atomic,
)
from imbalance_benchmark.modeling.workflows.tuning.tuning_reduction import ReduceRound


ScopeStream = tuple[Callable[[], Iterator[TuningScope]], int, list[dict[str, Any]]]


def run_candidate_shard(
    spec: ShardSpec,
    scopes: list[TuningScope],
    seeds: list[int],
    reduce_round: ReduceRound,
    output_root: Path,
    stage_one_config: dict[str, Any] | None = None,
    scope_stream: ScopeStream | None = None,
) -> Path:
    """Execute or reuse one candidate shard and persist its ordered observations."""
    complete = _reusable_path(spec, output_root, reduce_round)
    if complete is not None:
        return complete
    target = shard_path(output_root, spec)
    started_at, started = time.time(), time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    payload = (
        _fit_streamed_payload(spec, scope_stream[0], seeds, stage_one_config)
        if scope_stream
        else _fit_payload(spec, scopes, seeds, stage_one_config)
    )
    descriptors = scope_stream[2] if scope_stream else None
    payload["seeds"] = seeds
    payload["scope_count"] = scope_stream[1] if scope_stream else len(scopes)
    payload["observation_keys"] = _observation_keys(
        scopes, seeds, descriptors, spec.observation_index
    )
    payload.update(
        _runtime_payload(spec, reduce_round.fingerprint, started_at, started)
    )
    validate_shard_payload(payload, reduce_round.fingerprint, spec)
    _write_atomic(target, payload)
    return target


def _reusable_path(
    spec: ShardSpec, output_root: Path, reduce_round: ReduceRound
) -> Path | None:
    candidates = [spec]
    if spec.observation_index is not None:
        candidates.insert(0, replace(spec, observation_index=None))
    for candidate in candidates:
        path = shard_path(output_root, candidate)
        if path.exists():
            validate_shard_payload(
                json.loads(path.read_text()),
                reduce_round.fingerprint,
                candidate,
                reduce_round.accepted,
            )
            return path
    return None


def _runtime_payload(
    spec: ShardSpec,
    fingerprint: list[str],
    started_at: float,
    started: float,
) -> dict[str, Any]:
    elapsed = time.perf_counter() - started
    return {
        "complete": True,
        "fingerprint": fingerprint,
        "spec": asdict(spec),
        "started_at": started_at,
        "completed_at": time.time(),
        "accelerator_seconds": elapsed if torch.cuda.is_available() else 0.0,
        "peak_accelerator_memory_bytes": int(torch.cuda.max_memory_allocated())
        if torch.cuda.is_available()
        else 0,
        "hardware": {
            "cuda": torch.cuda.is_available(),
            "device": torch.cuda.get_device_name()
            if torch.cuda.is_available()
            else "cpu",
            "partition": os.environ.get("SLURM_JOB_PARTITION"),
            "job_id": os.environ.get("SLURM_JOB_ID"),
        },
    }


def _observation_keys(
    scopes: list[TuningScope],
    seeds: list[int],
    descriptors: list[dict[str, Any]] | None = None,
    observation_index: int | None = None,
) -> list[dict[str, Any]]:
    source = descriptors or [
        {
            "scope_index": scope_index,
            "assignment": scope.assignment,
            "split_index": scope.split_index,
        }
        for index, scope in enumerate(scopes)
        for scope_index in (getattr(scope, "scope_index", index),)
    ]
    keys = [
        {
            **scope,
            "seed_index": seed_index,
            "seed": seed,
        }
        for scope in source
        for seed_index, seed in enumerate(seeds)
    ]
    return (
        keys
        if observation_index is None
        else keys[observation_index : observation_index + 1]
    )


def _fit_streamed_payload(
    spec: ShardSpec,
    scope_provider: Callable[[], Iterator[TuningScope]],
    seeds: list[int],
    stage_one_config: dict[str, Any] | None,
) -> dict[str, Any]:
    """Run one feature-bank scope at a time in canonical observation order."""
    if spec.method == "post_hoc_logit_adjustment":
        if spec.observation_index is not None:
            raise RuntimeError("Post-hoc tuning must reduce all observations together")
        return _post_hoc_payload(scope_provider, seeds, stage_one_config)
    metrics: list[dict[str, Any]] = []
    cost_records: list[dict[str, int]] = []
    config: dict[str, Any] | None = None
    for scope in scope_provider():
        config = scope.regime.method_grids[spec.method][spec.candidate_index]
        scope.cost_records = cost_records
        for seed_index, seed in enumerate(seeds):
            index = scope.scope_index * len(seeds) + seed_index
            if spec.observation_index is not None and index != spec.observation_index:
                continue
            _, result = _evaluate(spec.method, config, scope, seed, stage_one_config)
            metrics.append(_observation_metric(scope, seed_index, seed, result))
    if config is None:
        raise RuntimeError("Tuning shard has no scopes")
    return {
        "candidate_index": spec.candidate_index,
        "config": config,
        "metrics": metrics,
        "cost_records": cost_records,
    }


def _post_hoc_payload(
    scope_source: list[TuningScope] | Callable[[], Iterator[TuningScope]],
    seeds: list[int],
    stage_one_config: dict[str, Any] | None,
) -> dict[str, Any]:
    """Select one post-hoc strength; a callable source streams scopes live
    (never materialize) rather than an already bank-resident base-phase list."""
    if stage_one_config is None:
        raise RuntimeError("Post-hoc tuning requires the selected CE configuration")
    cost_records: list[dict[str, int]] = []

    def _tagged() -> Iterator[TuningScope]:
        source = scope_source() if callable(scope_source) else scope_source
        for scope in source:
            scope.cost_records = cost_records
            yield scope

    return {
        "candidate_index": 0,
        "selection": _select_post_hoc(stage_one_config, _tagged(), seeds),
        "cost_records": cost_records,
    }


def _fit_payload(
    spec: ShardSpec,
    scopes: list[TuningScope],
    seeds: list[int],
    stage_one_config: dict[str, Any] | None,
) -> dict[str, Any]:
    if spec.method == "post_hoc_logit_adjustment":
        if spec.observation_index is not None:
            raise RuntimeError("Post-hoc tuning must reduce all observations together")
        return _post_hoc_payload(scopes, seeds, stage_one_config)
    config = scopes[0].regime.method_grids[spec.method][spec.candidate_index]
    metrics = []
    for scope in scopes:
        for seed_index, seed in enumerate(seeds):
            observation_index = scope.scope_index * len(seeds) + seed_index
            if (
                spec.observation_index is not None
                and observation_index != spec.observation_index
            ):
                continue
            _, result = _evaluate(spec.method, config, scope, seed, stage_one_config)
            metrics.append(_observation_metric(scope, seed_index, seed, result))
    return {
        "candidate_index": spec.candidate_index,
        "config": config,
        "metrics": metrics,
        "cost_records": scopes[0].cost_records,
    }


def _observation_metric(
    scope: TuningScope, seed_index: int, seed: int, result: dict[str, Any]
) -> dict[str, Any]:
    return {
        "scope_index": scope.scope_index,
        "seed_index": seed_index,
        "seed": seed,
        **{
            name: float(result[name])
            for name in ("balanced_accuracy", "macro_f1", "nll")
        },
    }
