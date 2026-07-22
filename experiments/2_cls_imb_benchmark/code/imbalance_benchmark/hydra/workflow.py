from __future__ import annotations

import argparse
from dataclasses import replace
from functools import partial
import logging
import subprocess
from typing import Any, Callable

from imbalance_benchmark.common import load_config
from imbalance_benchmark.hydra.rendering import SlurmJob, render_sbatch
from imbalance_benchmark.hydra.resume import verify_resume_freezes
from imbalance_benchmark.modeling.context import CONDITIONS
from imbalance_benchmark.modeling.context import roster_for_regime
from imbalance_benchmark.modeling.workflows.tuning.tuning_schedule import (
    DEPENDENT_METHODS,
    bundled_array_size,
    bundled_observation_array_size,
    candidate_array_size,
)

logger = logging.getLogger(__name__)


def _resources(
    config: dict[str, Any], stage: str, gpu: bool, fallback: str | None = None
) -> dict[str, Any]:
    sl = config.get("slurm", {})
    resources = sl.get("resources", {})
    sr = resources.get(stage, resources.get(fallback, {}))
    part = sr.get("partition", sl.get("partition", "gpu-2h" if gpu else "cpu-2h"))
    return {
        "partition": part,
        "gpus": int(sr.get("gpus", 1 if gpu else 0)),
        "cpus": int(sr.get("cpus", 4)),
        "memory": str(sr["memory"]) if sr.get("memory") else None,
        "time_limit": str(sr["time"]) if sr.get("time") else None,
    }


def _job(
    config: dict[str, Any],
    stage: str,
    cmd: str,
    gpu: bool,
    dependencies: tuple[str, ...] = (),
    resource: str | None = None,
    fallback: str | None = None,
) -> SlurmJob:
    return SlurmJob(
        stage,
        cmd,
        dependencies=dependencies,
        **_resources(config, resource or stage, gpu, fallback),
    )


def build_workflow(
    config: dict[str, Any], smoke: bool = False, resume_tuning: bool = False
) -> list[SlurmJob]:
    """Build the benchmark DAG, or its test-partition synthetic smoke variant."""
    if smoke:
        res = _resources(config, "smoke", True)
        res["partition"] = config.get("slurm", {}).get("test_partition", "gpu-test")
        return [SlurmJob("smoke", "smoke", **res)]
    arr = (0, 1, 2)
    setup = _setup_jobs(config, arr) if not resume_tuning else []
    freeze_dependency = ("freeze",) if setup else ()
    tuning = _tuning_jobs(config, freeze_dependency)
    co = replace(
        _job(config, "confirm", "confirm", True, ("tune-final-reduce",)),
        array_splits=arr,
        array_conditions=CONDITIONS,
    )
    an = _job(config, "analyze", "analyze", False, (co.name,))
    return [*setup, *tuning, co, an]


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
    config: dict[str, Any], freeze_dependency: tuple[str, ...]
) -> list[SlurmJob]:
    is_mil = config.get("dataset", {}).get("regime", "patch") == "wsi"
    roster = roster_for_regime(is_mil)
    natural_observations = int(
        config.get("slurm", {}).get("tune_natural_observations_per_candidate", 1)
    )
    shards_per_task = int(config.get("slurm", {}).get("tune_shards_per_task", 1))
    bundle_arg = f" --shards-per-task {shards_per_task}"
    bundle_size = partial(bundled_array_size, shards_per_task=shards_per_task)
    base_methods = tuple(method for method in roster if method not in DEPENDENT_METHODS)
    dependent_methods = tuple(
        method for method in roster if method in DEPENDENT_METHODS
    )
    base_natural = replace(
        _job(
            config,
            "tune-base-natural",
            "tune-shard --phase base --group natural"
            f" --observations-per-candidate {natural_observations}"
            f" --bundle-by-observation{bundle_arg}",
            True,
            freeze_dependency,
            "tune_natural",
            "tune",
        ),
        array_size=bundled_observation_array_size(
            candidate_array_size(base_methods), natural_observations, shards_per_task
        ),
    )
    base_controlled = replace(
        _job(
            config,
            "tune-base-controlled",
            f"tune-shard --phase base --group controlled{bundle_arg}",
            True,
            freeze_dependency,
            "tune_controlled",
            "tune",
        ),
        array_size=bundle_size(3 * candidate_array_size(base_methods)),
    )
    base_reduce = _job(
        config,
        "tune-base-reduce",
        "tune-reduce --phase base",
        False,
        (base_natural.name, base_controlled.name),
        "tune_reduce",
    )
    dependent_posthoc_natural = _job(
        config,
        "tune-dependent-posthoc-natural",
        "tune-shard --phase dependent --group natural --shard-index 0",
        True,
        (base_reduce.name,),
        "tune_post_hoc_natural",
        "tune_natural",
    )
    dependent_crt_natural = replace(
        _job(
            config,
            "tune-dependent-crt-natural",
            "tune-shard --phase dependent --group natural"
            f" --observations-per-candidate {natural_observations}"
            f" --shard-offset 1 --bundle-by-observation{bundle_arg}",
            True,
            (base_reduce.name,),
            "tune_natural",
            "tune",
        ),
        array_size=bundled_observation_array_size(
            candidate_array_size(("crt",)), natural_observations, shards_per_task
        ),
    )
    dependent_controlled = replace(
        _job(
            config,
            "tune-dependent-controlled",
            f"tune-shard --phase dependent --group controlled{bundle_arg}",
            True,
            (base_reduce.name,),
            "tune_controlled",
            "tune",
        ),
        array_size=bundle_size(3 * candidate_array_size(dependent_methods)),
    )
    final_reduce = _job(
        config,
        "tune-final-reduce",
        "tune-reduce --phase final",
        False,
        (
            dependent_posthoc_natural.name,
            dependent_crt_natural.name,
            dependent_controlled.name,
        ),
        "tune_reduce",
    )
    return [
        base_natural,
        base_controlled,
        base_reduce,
        dependent_posthoc_natural,
        dependent_crt_natural,
        dependent_controlled,
        final_reduce,
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
    submit: Callable[[str, bool], str] = _submit_script,
) -> dict[str, str]:
    """Render and submit the workflow in topological order, returning job IDs by stage."""
    if resume_tuning:
        verify_resume_freezes(config)
    submitted: dict[str, str] = {}
    for job in build_workflow(config, smoke, resume_tuning):
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
    submit_workflow(
        config,
        args.config,
        args.dry_run,
        getattr(args, "smoke", False),
        getattr(args, "resume_tuning", False),
    )
