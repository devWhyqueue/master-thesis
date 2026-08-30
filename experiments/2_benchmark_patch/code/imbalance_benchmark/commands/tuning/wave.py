"""Artifact-driven, capacity-bounded tuning-wave submission."""

from __future__ import annotations

import argparse
import logging
from contextlib import contextmanager, suppress
from dataclasses import replace
import os
from pathlib import Path
import time
from typing import Any, Iterator

from imbalance_benchmark.common import ensure_dirs, load_config
from imbalance_benchmark.commands.tuning.wave_round import (
    run_group_wave,
    run_round_wave,
)
from imbalance_benchmark.commands.tuning.wave_support import (
    record_attempted,
    submit_wave,
    unattempted,
)
from imbalance_benchmark.hydra.queue import DEFAULT_QUEUE_CAP, _squeue_count
from imbalance_benchmark.hydra.rendering import render_sbatch
from imbalance_benchmark.hydra.rendering import SlurmJob
from imbalance_benchmark.hydra.resume import ResumePlan, resume_plan
from imbalance_benchmark.hydra.workflow import _submit_script, _tuning_jobs
from imbalance_benchmark.modeling.context import CONDITIONS
from imbalance_benchmark.modeling.workflows.tuning.aggregation.candidate_registry import (
    round_state_path,
)

__all__ = ["cmd_tune_wave", "select_wave"]

logger = logging.getLogger(__name__)

LOCK_POLL_SECONDS = 15
LOCK_WAIT_SECONDS = 900
# A submission holds the lock for seconds; anything older lost its job.
LOCK_STALE_SECONDS = 1800


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


def _reclaim_stale_lock(path: Path) -> bool:
    """Report whether the lock is free again, clearing one a killed job left."""
    try:
        age = time.time() - path.stat().st_mtime
    except FileNotFoundError:
        return True
    if age < LOCK_STALE_SECONDS:
        return False
    logger.warning("Reclaiming tuning submission lock held for %.0fs: %s", age, path)
    try:
        path.rmdir()
    except OSError:
        return False
    return True


@contextmanager
def _submission_lock(path: Path) -> Iterator[None]:
    """Serialize shared-output wave submissions on BeeGFS.

    Sibling conditions routinely reach their wave submission within the same
    minute, so a busy lock is expected rather than exceptional. Wait for the
    holder instead of failing: nothing retries a crashed wave submission, so
    a lost race silently freezes that condition's round chain forever.
    """
    deadline = time.monotonic() + LOCK_WAIT_SECONDS
    while True:
        try:
            path.mkdir()
            break
        except FileExistsError as error:
            if _reclaim_stale_lock(path):
                continue
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"Tuning submission lock still busy after {LOCK_WAIT_SECONDS}s: "
                    f"{path}. Resume with: submit --resume-tuning"
                ) from error
            time.sleep(LOCK_POLL_SECONDS)
    try:
        yield
    finally:
        with suppress(FileNotFoundError):
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
        _submit_terminal(config, os.path.abspath(args.config), base)
        return
    fresh = ResumePlan(
        tuple(
            unattempted(base["data"], "base-natural", list(remaining.natural_indices))
        ),
        tuple(
            unattempted(
                base["data"], "base-controlled", list(remaining.controlled_indices)
            )
        ),
    )
    if not fresh.natural_indices and not fresh.controlled_indices:
        logger.warning(
            "tune-wave base: %d natural + %d controlled shard(s) still missing after "
            "a prior attempt; not retrying. Investigate the failed tasks, then "
            "resume with: submit --resume-tuning",
            len(remaining.natural_indices),
            len(remaining.controlled_indices),
        )
        return
    limit = int(config.get("slurm", {}).get("tuning_wave_task_limit", 90))
    wave = select_wave(fresh, _squeue_count(), limit)
    record_attempted(base["data"], "base-natural", list(wave.natural_indices))
    record_attempted(base["data"], "base-controlled", list(wave.controlled_indices))
    submit_wave(config, os.path.abspath(args.config), _wave_jobs(config, wave), args)


def _submit_terminal(
    config: dict[str, Any], config_path: str, base: dict[str, Path]
) -> None:
    """Submit reducer and decision jobs only after every validated base shard exists.

    A condition whose round-0 decide already ran has moved into its own
    self-chained round or dependent-phase wave, so this resume path must
    never resubmit its ``tune-decide-base-*`` round 0: that would recompute
    round 0's decision from scratch and clobber the further-along state a
    later round already locked in, corrupting candidate-registry indexing.
    """
    decided = {
        condition
        for condition in CONDITIONS
        if round_state_path(base["data"], condition).exists()
    }
    jobs = [
        job
        for job in _tuning_jobs(config, (), ResumePlan((), ()))
        if job.name not in {f"tune-decide-base-{condition}" for condition in decided}
    ]
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
