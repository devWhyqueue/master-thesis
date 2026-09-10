"""SLURM submission for the patient-influence experiment's preflight -> fit -> analyze DAG."""

from __future__ import annotations

import logging
import subprocess
from dataclasses import replace
from typing import Any, Callable

from imbalance_benchmark.hydra.guards import check_queue_cap
from imbalance_benchmark.hydra.job_resources import resources_for
from imbalance_benchmark.hydra.rendering import SlurmJob, render_sbatch

from influence.fit import FIT_SHARD_COUNT

__all__ = ["build_workflow", "submit_workflow"]

logger = logging.getLogger(__name__)


def _job(
    config: dict[str, Any],
    stage: str,
    command: str,
    dependencies: tuple[str, ...] = (),
    array_size: int = 0,
) -> SlurmJob:
    """Build one stage's job with its resolved SLURM resources."""
    res = resources_for(config, stage, False)
    return SlurmJob(
        stage, command, dependencies=dependencies, array_size=array_size, **res
    )


def build_workflow(config: dict[str, Any]) -> list[SlurmJob]:
    """Build the full DAG: preflight -> fit (array 0-5) -> analyze."""
    preflight = _job(config, "preflight", "preflight")
    fit = _job(
        config, "fit", "fit", dependencies=(preflight.name,), array_size=FIT_SHARD_COUNT
    )
    analyze = _job(config, "analyze", "analyze", dependencies=(fit.name,))
    return [preflight, fit, analyze]


def _submit_script(script: str, dry_run: bool) -> str:
    del dry_run
    result = subprocess.run(
        ["sbatch", "--parsable"],
        input=script,
        text=True,
        check=True,
        capture_output=True,
    )
    return result.stdout.strip().split(";", maxsplit=1)[0]


def submit_workflow(
    config: dict[str, Any],
    config_path: str | None = None,
    dry_run: bool = False,
    submit: Callable[[str, bool], str] = _submit_script,
) -> dict[str, str]:
    """Render and submit the DAG jobs in dependency order."""
    submitted: dict[str, str] = {}
    for job in build_workflow(config):
        dependencies = tuple(submitted[name] for name in job.dependencies)
        scheduled = replace(job, dependencies=dependencies)
        script = render_sbatch(scheduled, config, config_path)
        if dry_run:
            jid = f"dry-run-{job.name}"
            logger.info("%s", script)
        else:
            check_queue_cap()
            jid = submit(script, False)
        submitted[job.name] = jid
        logger.info("%s: %s", job.name, jid)
    return submitted
