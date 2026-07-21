from __future__ import annotations

from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import time
from typing import Any

import torch

from imbalance_benchmark.datasets.data import load_training_dataset
from imbalance_benchmark.modeling.workflows.tuning_aggregate import (
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


def combined_scopes(
    raw_scopes: list[tuple[dict[str, Path], Any, torch.utils.data.DataLoader]],
    condition: str,
    assignments: tuple[str, ...],
    cost_records: list[dict[str, int]] | None = None,
) -> list[TuningScope]:
    """Build the canonical assignment-then-split tuning observations."""
    records = cost_records if cost_records is not None else []
    result = []
    for assignment in assignments:
        for split_index, (paths, regime, loader) in enumerate(raw_scopes):
            manifest = _manifest_name(condition, assignment)
            result.append(
                TuningScope(
                    regime,
                    loader,
                    load_training_dataset(
                        paths["data"] / manifest,
                        regime.is_mil,
                        class_names=regime.locked_class_names,
                        bag_kwargs=regime.bag_dataset_kwargs,
                    ),
                    records,
                    regime.update_budgets.get(
                        "natural" if condition == "natural" else "controlled"
                    ),
                    assignment,
                    split_index,
                )
            )
    return result


def _manifest_name(condition: str, assignment: str) -> str:
    return (
        f"manifest_{condition}.csv"
        if condition in {"natural", "balanced"}
        else f"manifest_{assignment}_{condition}.csv"
    )


def run_candidate_shard(
    spec: ShardSpec,
    scopes: list[TuningScope],
    seeds: list[int],
    fingerprint: list[str],
    output_root: Path,
    stage_one_config: dict[str, Any] | None = None,
) -> Path:
    """Execute or reuse one candidate shard and persist its ordered observations."""
    complete = _reusable_path(spec, output_root, fingerprint)
    if complete is not None:
        return complete
    target = shard_path(output_root, spec)
    started_at, started = time.time(), time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    payload = _fit_payload(spec, scopes, seeds, stage_one_config)
    observation_keys = _observation_keys(scopes, seeds)
    if spec.observation_index is not None:
        observation_keys = observation_keys[
            spec.observation_index : spec.observation_index + 1
        ]
    payload.update(
        {
            "seeds": seeds,
            "scope_count": len(scopes),
            "observation_keys": observation_keys,
        }
    )
    payload.update(_runtime_payload(spec, fingerprint, started_at, started))
    validate_shard_payload(payload, fingerprint, spec)
    _write_atomic(target, payload)
    return target


def _reusable_path(
    spec: ShardSpec, output_root: Path, fingerprint: list[str]
) -> Path | None:
    candidates = [spec]
    if spec.observation_index is not None:
        candidates.insert(0, replace(spec, observation_index=None))
    for candidate in candidates:
        path = shard_path(output_root, candidate)
        if path.exists():
            validate_shard_payload(json.loads(path.read_text()), fingerprint, candidate)
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
        "hardware": _hardware(),
    }


def _hardware() -> dict[str, Any]:
    return {
        "cuda": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "job_id": os.environ.get("SLURM_JOB_ID"),
    }


def _observation_keys(
    scopes: list[TuningScope], seeds: list[int]
) -> list[dict[str, Any]]:
    return [
        {
            "scope_index": scope_index,
            "assignment": scope.assignment,
            "split_index": scope.split_index,
            "seed_index": seed_index,
            "seed": seed,
        }
        for scope_index, scope in enumerate(scopes)
        for seed_index, seed in enumerate(seeds)
    ]


def _fit_payload(
    spec: ShardSpec,
    scopes: list[TuningScope],
    seeds: list[int],
    stage_one_config: dict[str, Any] | None,
) -> dict[str, Any]:
    if spec.method == "post_hoc_logit_adjustment":
        if spec.observation_index is not None:
            raise RuntimeError("Post-hoc tuning must reduce all observations together")
        if stage_one_config is None:
            raise RuntimeError("Post-hoc tuning requires the selected CE configuration")
        selected = _select_post_hoc(stage_one_config, scopes, seeds)
        return {
            "candidate_index": 0,
            "selection": selected,
            "cost_records": scopes[0].cost_records,
        }
    config = scopes[0].regime.method_grids[spec.method][spec.candidate_index]
    metrics = []
    for scope_index, scope in enumerate(scopes):
        for seed_index, seed in enumerate(seeds):
            observation_index = scope_index * len(seeds) + seed_index
            if (
                spec.observation_index is not None
                and observation_index != spec.observation_index
            ):
                continue
            _, result = _evaluate(spec.method, config, scope, seed, stage_one_config)
            metrics.append(
                {
                    "scope_index": scope_index,
                    "seed_index": seed_index,
                    "seed": seed,
                    **{
                        name: float(result[name])
                        for name in ("balanced_accuracy", "macro_f1", "nll")
                    },
                }
            )
    return {
        "candidate_index": spec.candidate_index,
        "config": config,
        "metrics": metrics,
        "cost_records": scopes[0].cost_records,
    }
