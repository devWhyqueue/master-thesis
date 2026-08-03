from __future__ import annotations

import argparse
from dataclasses import replace
import logging
import os
import subprocess
from typing import Any, Callable

from imbalance_benchmark.common import load_config
from imbalance_benchmark.hydra.confirm_jobs import confirm_jobs as _confirm_jobs
from imbalance_benchmark.hydra.job_resources import build_job as _job
from imbalance_benchmark.hydra.job_resources import resources_for as _resources
from imbalance_benchmark.hydra.rendering import SlurmJob, render_sbatch
from imbalance_benchmark.hydra.resume import ResumePlan, resume_plan
from imbalance_benchmark.modeling.context import CONDITIONS
from imbalance_benchmark.modeling.workflows.tuning.tuning_schedule import (
    bundled_array_size,
    bundled_observation_array_size,
    candidate_array_size,
    phase_methods,
)

logger = logging.getLogger(__name__)


def _analyze_jobs(
    config: dict[str, Any],
    confirm_natural: SlurmJob,
    confirm_controlled: SlurmJob,
    arr: tuple[int, ...],
) -> tuple[SlurmJob, SlurmJob]:
    """The 3-way analyze array plus its dependent equal-split aggregation job."""
    an = replace(
        _job(
            config,
            "analyze",
            "analyze",
            False,
            (confirm_natural.name, confirm_controlled.name),
        ),
        array_splits=arr,
    )
    return an, _job(
        config, "analyze-combine", "analyze-combine", False, (an.name,), "analyze"
    )


def build_workflow(
    config: dict[str, Any],
    smoke: bool = False,
    resume_tuning: bool = False,
    confirm_only: bool = False,
) -> list[SlurmJob]:
    """Build the benchmark DAG, or its test-partition synthetic smoke variant.

    Tuning's adaptive search rounds submit themselves once a prior round
    completes (see ``tune-decide``), so their true finish time is unknown at
    submit time - confirm and analyze can never be statically chained after
    them. ``confirm_only`` builds just that later stage, submitted
    separately once every condition's tuning lock is resolved.
    """
    if smoke:
        res = _resources(config, "smoke", True)
        res["partition"] = config.get("slurm", {}).get("test_partition", "gpu-test")
        return [SlurmJob("smoke", "smoke", **res)]
    arr = (0, 1, 2)
    if confirm_only:
        confirm_natural, confirm_controlled = _confirm_jobs(config)
        an, combine = _analyze_jobs(config, confirm_natural, confirm_controlled, arr)
        return [confirm_natural, confirm_controlled, an, combine]
    setup = _setup_jobs(config, arr) if not resume_tuning else []
    freeze_dependency = ("freeze",) if setup else ()
    plan = resume_plan(config) if resume_tuning else None
    return [*setup, *_tuning_jobs(config, freeze_dependency, plan)]


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
    return [prepare, pilot, freeze]


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
                3 * candidate_array_size(controlled_methods), controlled_shards
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
    smoke: bool = False,
    resume_tuning: bool = False,
    confirm_only: bool = False,
    submit: Callable[[str, bool], str] = _submit_script,
) -> dict[str, str]:
    """Render and submit the workflow in topological order, returning job IDs by stage."""
    submitted: dict[str, str] = {}
    for job in build_workflow(config, smoke, resume_tuning, confirm_only):
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
    submit_workflow(
        config,
        config_path,
        args.dry_run,
        getattr(args, "smoke", False),
        getattr(args, "resume_tuning", False),
        getattr(args, "confirm_only", False),
    )
