from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import logging
import os
import subprocess
from typing import Any, Callable

from imbalance_benchmark.common import load_config
from imbalance_benchmark.hydra.confirm_jobs import confirm_jobs
from imbalance_benchmark.hydra.job_resources import build_job as _job
from imbalance_benchmark.hydra.job_resources import resources_for
from imbalance_benchmark.hydra.job_resources import stage_jobs
from imbalance_benchmark.hydra.rendering import SlurmJob, render_sbatch
from imbalance_benchmark.hydra.resume import ResumePlan
from imbalance_benchmark.modeling.context import CONDITIONS, CONTROLLED_CONDITIONS
from imbalance_benchmark.modeling.workflows.tuning.tuning_schedule import (
    bundled_array_size,
    bundled_observation_array_size,
    candidate_array_size,
    phase_methods,
)

logger = logging.getLogger(__name__)


def smoke_workflow(config: dict[str, Any]) -> list[SlurmJob]:
    """Build the single test-partition smoke allocation."""
    resources = resources_for(config, "smoke", True)
    resources["partition"] = config.get("slurm", {}).get("test_partition", "gpu-test")
    return [SlurmJob("smoke", "smoke", **resources)]


def confirm_workflow(config: dict[str, Any]) -> list[SlurmJob]:
    """Build confirmation and its later analysis DAG."""
    natural, controlled = confirm_jobs(config)
    analyze = replace(
        _job(config, "analyze", "analyze", False, (natural.name, controlled.name)),
        array_splits=(0, 1, 2),
    )
    combine = _job(
        config, "analyze-combine", "analyze-combine", False, (analyze.name,), "analyze"
    )
    return [natural, controlled, analyze, combine]


def resume_tuning_job(config: dict[str, Any]) -> SlurmJob:
    """Build one artifact-driven tuning-wave controller."""
    return _job(
        config,
        "tune-wave",
        "tune-wave",
        False,
        resource="tune_decide",
        fallback="tune_reduce",
    )


@dataclass(frozen=True)
class SubmitOptions:
    """Optional workflow mode and one explicit stage boundary."""

    smoke: bool = False
    resume_tuning: bool = False
    confirm_only: bool = False
    stage: str | None = None
    split_index: int | None = None


def build_workflow(
    config: dict[str, Any],
    smoke: bool = False,
    resume_tuning: bool = False,
    confirm_only: bool = False,
    stage: str | None = None,
    split_index: int | None = None,
) -> list[SlurmJob]:
    """Build the benchmark DAG, or its test-partition synthetic smoke variant.

    Tuning's adaptive search rounds submit themselves once a prior round
    completes (see ``tune-decide``), so confirm/analyze can never be
    statically chained after them. ``confirm_only`` builds just that later
    stage, submitted separately once every condition's tuning lock resolves.
    """
    if smoke:
        return smoke_workflow(config)
    if stage:
        return stage_jobs(config, stage, split_index)
    if confirm_only:
        return confirm_workflow(config)
    if resume_tuning:
        return [resume_tuning_job(config)]
    setup = _setup_jobs(config, (0, 1, 2))
    freeze_dependency = ("freeze",) if setup else ()
    return [*setup, *_tuning_jobs(config, freeze_dependency, None)]


def _setup_jobs(config: dict[str, Any], splits: tuple[int, ...]) -> list[SlurmJob]:
    prepare = _job(config, "prepare", "prepare", True)
    pilot = replace(
        _job(config, "pilot", "pilot", False, (prepare.name,)),
        array_splits=splits,
    )
    freeze = replace(
        _job(config, "freeze", "freeze", False, (pilot.name,)),
        array_splits=splits,
    )
    signals = replace(
        _job(config, "signals", "signals", False, (freeze.name,)),
        array_splits=splits,
    )
    return [prepare, pilot, freeze, signals]


def _tuning_jobs(
    config: dict[str, Any], freeze_dependency: tuple[str, ...], plan: ResumePlan | None
) -> list[SlurmJob]:
    is_mil = config.get("dataset", {}).get("regime", "patch") == "wsi"
    natural_observations = int(
        config.get("slurm", {}).get("tune_natural_observations_per_candidate", 1)
    )
    slurm = config.get("slurm", {})
    natural_shards = int(
        slurm.get("tune_natural_shards_per_task", slurm.get("tune_shards_per_task", 1))
    )
    controlled_shards = int(slurm.get("tune_shards_per_task", 1))
    natural_methods = phase_methods(is_mil, "base", "natural")
    controlled_methods = phase_methods(is_mil, "base", "balanced")
    base_natural = (
        replace(
            _job(
                config,
                "tune-base-natural",
                "tune-shard --phase base --group natural"
                f" --observations-per-candidate {natural_observations}"
                f" --bundle-by-observation --shards-per-task {natural_shards}",
                True,
                freeze_dependency,
                "tune_natural",
                "tune",
            ),
            array_size=bundled_observation_array_size(
                candidate_array_size(natural_methods),
                natural_observations,
                natural_shards,
            ),
            array_indices=plan.natural_indices if plan else (),
        )
        if not plan or plan.natural_indices
        else None
    )
    base_controlled = (
        replace(
            _job(
                config,
                "tune-base-controlled",
                "tune-shard --phase base --group controlled"
                f" --shards-per-task {controlled_shards}",
                True,
                freeze_dependency,
                "tune_controlled",
                "tune",
            ),
            array_size=bundled_array_size(
                len(CONTROLLED_CONDITIONS) * candidate_array_size(controlled_methods),
                controlled_shards,
            ),
            array_indices=plan.controlled_indices if plan else (),
        )
        if not plan or plan.controlled_indices
        else None
    )
    base_reduce = _job(
        config,
        "tune-base-reduce",
        "tune-reduce --phase base",
        False,
        tuple(job.name for job in (base_natural, base_controlled) if job),
        "tune_reduce",
    )
    decisions = [
        _job(
            config,
            f"tune-decide-base-{condition}",
            f"tune-decide --phase base --condition {condition} --round 0",
            False,
            (base_reduce.name,),
            "tune_decide",
            "tune_reduce",
        )
        for condition in CONDITIONS
    ]
    return [
        *[job for job in (base_natural, base_controlled) if job],
        base_reduce,
        *decisions,
    ]


def _submit_script(script: str, dry_run: bool) -> str:
    if dry_run:
        return "dry-run"
    cmd = ["sbatch", "--parsable"]
    res = subprocess.run(cmd, input=script, text=True, check=True, capture_output=True)
    return res.stdout.strip().split(";", maxsplit=1)[0]


def submit_workflow(
    config: dict[str, Any],
    config_path: str | None = None,
    dry_run: bool = False,
    options: SubmitOptions = SubmitOptions(),
    submit: Callable[[str, bool], str] = _submit_script,
) -> dict[str, str]:
    """Render and submit the workflow in topological order, returning job IDs by stage."""
    submitted: dict[str, str] = {}
    for job in build_workflow(
        config,
        options.smoke,
        options.resume_tuning,
        options.confirm_only,
        options.stage,
        options.split_index,
    ):
        dependencies = tuple(submitted[name] for name in job.dependencies)
        scheduled = replace(job, dependencies=dependencies)
        script = render_sbatch(scheduled, config, config_path)
        jid = f"dry-run-{job.name}" if dry_run else submit(script, False)
        submitted[job.name] = jid
        logger.info("%s: %s", job.name, jid)
        if dry_run:
            logger.info("%s", script)
    return submitted


def cmd_submit(args: argparse.Namespace) -> None:
    """Submit the Hydra workflow."""
    config = load_config(args.config)
    # Baked into rendered sbatch scripts, which cd to the project root before
    # running — a relative --config path must be resolved before embedding.
    config_path = os.path.abspath(args.config)
    options = SubmitOptions(
        getattr(args, "smoke", False),
        getattr(args, "resume_tuning", False),
        getattr(args, "confirm_only", False),
        getattr(args, "stage", None),
        args.split_index,
    )
    submit_workflow(config, config_path, args.dry_run, options)
