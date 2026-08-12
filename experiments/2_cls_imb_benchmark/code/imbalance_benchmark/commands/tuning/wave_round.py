"""Adaptive and dependent tuning-wave artifact planning."""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace
import os
from pathlib import Path
from typing import Any

from imbalance_benchmark.commands.tuning import _frozen_shard_context
from imbalance_benchmark.commands.tuning.wave_support import (
    record_attempted,
    submit_wave,
    unattempted,
)
from imbalance_benchmark.hydra.dependent_jobs import dependent_round_zero_jobs
from imbalance_benchmark.hydra.job_resources import build_job
from imbalance_benchmark.hydra.queue import DEFAULT_QUEUE_CAP, _squeue_count
from imbalance_benchmark.hydra.rendering import SlurmJob, render_sbatch
from imbalance_benchmark.hydra.workflow import _submit_script
from imbalance_benchmark.modeling.workflows.tuning.tuning_artifacts import (
    expected_observations,
    load_candidate,
)
from imbalance_benchmark.modeling.workflows.tuning.tuning_schedule import (
    bundled_observation_array_size,
    candidate_array_size,
    phase_methods,
    requested_shard,
    resolve_round_shard_spec,
)

logger = logging.getLogger(__name__)


def run_round_wave(
    config: dict[str, Any], base: dict[str, Path], args: argparse.Namespace
) -> None:
    """Submit or finish one condition's adaptive or dependent round."""
    job, pending = _round_plan(config, args)
    if not pending:
        _submit_decide(config, os.path.abspath(args.config), args)
        return
    scope = f"round-{args.phase}-{args.condition}-{args.round}"
    fresh = unattempted(base["data"], scope, pending)
    if not fresh:
        _warn_stalled(scope, len(pending))
        return
    limited = _limited(job, fresh, config)
    record_attempted(base["data"], scope, list(limited.array_indices))
    _submit_wave(config, os.path.abspath(args.config), limited, args)


def run_group_wave(
    config: dict[str, Any], base: dict[str, Path], args: argparse.Namespace
) -> None:
    """Submit frozen dependent controlled group through same wave loop."""
    if args.phase != "dependent" or args.round:
        raise ValueError("Only dependent round zero accepts a tuning group")
    _, _, freeze, fingerprint = _frozen_shard_context(args, False)
    is_mil = config.get("dataset", {}).get("regime", "patch") == "wsi"
    (job,) = dependent_round_zero_jobs(config, is_mil)
    pending = [
        index
        for index in range(job.array_size)
        if _group_missing(base["data"], freeze, fingerprint, is_mil, job, index)
    ]
    if not pending:
        _submit_group_decides(config, os.path.abspath(args.config))
        return
    scope = "group-dependent"
    fresh = unattempted(base["data"], scope, pending)
    if not fresh:
        _warn_stalled(scope, len(pending))
        return
    limited = _limited(job, fresh, config)
    record_attempted(base["data"], scope, list(limited.array_indices))
    _submit_wave(config, os.path.abspath(args.config), limited, args)


def _warn_stalled(scope: str, missing: int) -> None:
    logger.warning(
        "tune-wave %s: %d shard(s) still missing after a prior attempt; not "
        "retrying. Investigate the failed tasks, then resume with: "
        "submit --resume-tuning",
        scope,
        missing,
    )


def _round_plan(
    config: dict[str, Any], args: argparse.Namespace
) -> tuple[SlurmJob, list[int]]:
    base, _, freeze, fingerprint = _frozen_shard_context(args, False)
    is_mil = config.get("dataset", {}).get("regime", "patch") == "wsi"
    methods = phase_methods(is_mil, args.phase, str(args.condition))
    job = _round_job(config, args, methods)
    assignments = tuple(freeze.get("tail_assignments", {"native": []}))
    expected = expected_observations(str(args.condition), assignments, freeze)
    observations = int(
        config.get("slurm", {}).get("tune_natural_observations_per_candidate", 1)
    )
    pending = [
        index
        for index in range(job.array_size)
        if _round_missing(
            base["data"], args, methods, index, fingerprint, expected, observations
        )
    ]
    return job, pending


def _round_job(
    config: dict[str, Any], args: argparse.Namespace, methods: tuple[str, ...]
) -> SlurmJob:
    name = f"tune-wave-{args.condition}-{args.phase}-r{args.round}"
    command = f"tune-shard --phase {args.phase} --condition {args.condition} --round {args.round}"
    candidates = candidate_array_size(methods)
    if args.condition != "natural":
        return replace(
            build_job(config, name, command, True, (), "tune_controlled", "tune"),
            array_size=candidates,
        )
    observations = int(
        config.get("slurm", {}).get("tune_natural_observations_per_candidate", 1)
    )
    command += f" --observations-per-candidate {observations} --bundle-by-observation --shards-per-task 1"
    return replace(
        build_job(
            config, name, command, True, (), "tune_natural_round", "tune_controlled"
        ),
        array_size=bundled_observation_array_size(candidates, observations, 1),
    )


def _round_missing(
    root: Path,
    args: argparse.Namespace,
    methods: tuple[str, ...],
    index: int,
    fingerprint: list[str],
    expected: int,
    observations: int,
) -> bool:
    candidate, observation = (
        divmod(index, observations) if args.condition == "natural" else (index, None)
    )
    spec = resolve_round_shard_spec(
        root, str(args.condition), candidate, args.phase, methods
    )
    if spec is None:
        return False
    try:
        load_candidate(
            root,
            replace(spec, observation_index=observation),
            fingerprint,
            None if observation is not None else expected,
        )
    except RuntimeError as error:
        if str(error).startswith("Missing tuning"):
            return True
        raise
    return False


def _group_missing(
    root: Path,
    freeze: dict[str, Any],
    fingerprint: list[str],
    is_mil: bool,
    job: SlurmJob,
    index: int,
) -> bool:
    assignments = tuple(freeze.get("tail_assignments", {"native": []}))
    shards = int(job.command.rsplit(" ", 1)[-1])
    for shard in range(index * shards, (index + 1) * shards):
        spec = requested_shard(
            shard, "dependent", "controlled", is_mil, freeze["method_grids"], None
        )
        if spec is None:
            continue
        try:
            load_candidate(
                root,
                spec,
                fingerprint,
                expected_observations(spec.condition, assignments, freeze),
            )
        except RuntimeError as error:
            if str(error).startswith("Missing tuning"):
                return True
            raise
    return False


def _limited(job: SlurmJob, pending: list[int], config: dict[str, Any]) -> SlurmJob:
    limit = int(config.get("slurm", {}).get("tuning_wave_task_limit", 90))
    available = min(limit, DEFAULT_QUEUE_CAP - _squeue_count() - 1)
    if available < 1:
        raise RuntimeError(
            "No safe tuning wave fits. Resume with: submit --resume-tuning"
        )
    return replace(job, array_indices=tuple(pending[:available]))


def _submit_wave(
    config: dict[str, Any],
    config_path: str,
    job: SlurmJob,
    args: argparse.Namespace,
) -> None:
    submit_wave(config, config_path, [job], args)


def _submit_decide(
    config: dict[str, Any], config_path: str, args: argparse.Namespace
) -> None:
    job = build_job(
        config,
        f"tune-decide-{args.condition}-{args.phase}-r{args.round}",
        f"tune-decide --phase {args.phase} --condition {args.condition} --round {args.round}",
        False,
        (),
        "tune_decide",
        "tune_reduce",
    )
    _submit_script(render_sbatch(job, config, config_path), False)


def _submit_group_decides(config: dict[str, Any], config_path: str) -> None:
    for condition in ("balanced", "moderate", "severe"):
        _submit_decide(
            config,
            config_path,
            argparse.Namespace(phase="dependent", condition=condition, round=0),
        )
