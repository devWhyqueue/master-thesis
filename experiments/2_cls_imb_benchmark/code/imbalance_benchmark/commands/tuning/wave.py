"""Artifact-driven, capacity-bounded tuning-wave submission."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import replace
import os
from pathlib import Path
from typing import Any, Iterator

from imbalance_benchmark.common import ensure_dirs, load_config
from imbalance_benchmark.commands.tuning.wave_round import (
    run_group_wave,
    run_round_wave,
)
from imbalance_benchmark.commands.tuning.wave_support import next_stalled, submit_wave
from imbalance_benchmark.hydra.queue import DEFAULT_QUEUE_CAP, _squeue_count
from imbalance_benchmark.hydra.rendering import render_sbatch
from imbalance_benchmark.hydra.rendering import SlurmJob
from imbalance_benchmark.hydra.resume import ResumePlan, resume_plan
from imbalance_benchmark.hydra.workflow import _submit_script, _tuning_jobs

__all__ = ["cmd_tune_wave", "select_wave"]


def select_wave(plan: ResumePlan, queued: int, limit: int) -> ResumePlan:
    """Select original sparse task IDs, natural first, reserving successor space."""
    available = min(limit, DEFAULT_QUEUE_CAP - queued - 1)
    if available < 1:
        raise RuntimeError(
            f"Queue at {queued} projected tasks (cap {DEFAULT_QUEUE_CAP}); no safe "
            "tuning wave fits. Resume with: submit --resume-tuning"
        )
    natural = plan.natural_indices[:available]
    controlled = plan.controlled_indices[: available - len(natural)]
    return ResumePlan(natural, controlled)


@contextmanager
def _submission_lock(path: Path) -> Iterator[None]:
    """Serialize shared-output wave submissions on BeeGFS."""
    try:
        path.mkdir()
    except FileExistsError as error:
        raise RuntimeError(f"Tuning submission lock is busy: {path}") from error
    try:
        yield
    finally:
        path.rmdir()


def _wave_jobs(config: dict[str, Any], plan: ResumePlan | SlurmJob) -> list[SlurmJob]:
    """Return only this wave's sparse array jobs; reducer waits for final wave."""
    if isinstance(plan, SlurmJob):
        return [plan]
    return [
        job
        for job in _tuning_jobs(config, (), plan)
        if job.name in {"tune-base-natural", "tune-base-controlled"}
    ]


def cmd_tune_wave(args: argparse.Namespace) -> None:
    """Submit one base tuning wave and one afterany successor under queue cap."""
    config = load_config(args.config)
    base = ensure_dirs(config)
    with _submission_lock(base["data"] / ".tune-wave.lock"):
        if getattr(args, "group", None) is not None:
            run_group_wave(config, base, args)
        elif args.condition is None:
            _run_base_wave(config, base, args)
        else:
            run_round_wave(config, base, args)


def _run_base_wave(
    config: dict[str, Any], base: dict[str, Path], args: argparse.Namespace
) -> None:
    """Submit or finish the initial frozen-grid wave."""
    if args.phase != "base" or args.round:
        raise ValueError("A condition is required for non-base tuning waves")
    remaining = resume_plan(config)
    if not remaining.natural_indices and not remaining.controlled_indices:
        _submit_terminal(config, os.path.abspath(args.config))
        return
    pending = [*remaining.natural_indices, *remaining.controlled_indices]
    stalled = next_stalled(base["data"], pending, args.stalled_waves)
    limit = int(config.get("slurm", {}).get("tuning_wave_task_limit", 90))
    submit_wave(
        config,
        os.path.abspath(args.config),
        _wave_jobs(config, select_wave(remaining, _squeue_count(), limit)),
        args,
        stalled,
    )


def _submit_terminal(config: dict[str, Any], config_path: str) -> None:
    """Submit reducer and decision jobs only after every validated base shard exists."""
    jobs = _tuning_jobs(config, (), ResumePlan((), ()))
    if _squeue_count() + len(jobs) > DEFAULT_QUEUE_CAP:
        raise RuntimeError(
            "No room for tuning terminal jobs. Resume with: submit --resume-tuning"
        )
    submitted: dict[str, str] = {}
    for job in jobs:
        scheduled = replace(
            job, dependencies=tuple(submitted[name] for name in job.dependencies)
        )
        submitted[job.name] = _submit_script(
            render_sbatch(scheduled, config, config_path), False
        )
