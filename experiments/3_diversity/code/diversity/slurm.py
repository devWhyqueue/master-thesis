"""SLURM submission: exp-3's own two-phase DAG, rendered with exp-2's SlurmJob machinery.

Reuses ``hydra/rendering.py::SlurmJob`` and ``render_sbatch`` directly, and
copies (does not import) the ~12-line submit loop from
``hydra/workflow.py::submit_workflow`` -- that function hard-codes exp-2's
own DAG, so only the loop pattern is reused (plan "Cluster wiring").

Two phases, matching the plan's "Execution order" (build/check first, a
human reviews Gate 0, then fit/analyze):

- ``build``: ``build`` (array over splits 0-2) -> ``check`` (one job).
- ``fit``: ``import-anchor`` -> ``fit-standard`` + ``fit-semantic-scale``
  (both arrays) -> ``analyze``.

Nothing here submits a job on its own import; :func:`submit_workflow` is
the only function that shells out to ``sbatch``, and never runs in this
task (no cluster access).
"""

from __future__ import annotations

import logging
import math
import subprocess
from dataclasses import replace
from typing import Any, Callable

from imbalance_benchmark.hydra.guards import check_queue_cap
from imbalance_benchmark.hydra.job_resources import resources_for
from imbalance_benchmark.hydra.rendering import SlurmJob, render_sbatch

from diversity.fit import fit_units

__all__ = ["build_workflow", "submit_workflow"]

logger = logging.getLogger(__name__)

# Defaults if a config omits its own knob; 'semantic_scale' carries an
# SsbPool re-encode cost (plan "Sharding"), so it packs fewer items/task.
_DEFAULT_SHARDS_PER_TASK = {"standard": 6, "semantic_scale": 2}


def _job(
    config: dict[str, Any],
    stage: str,
    command: str,
    gpu: bool,
    dependencies: tuple[str, ...] = (),
    resource: str | None = None,
) -> SlurmJob:
    """Build one stage's job with its resolved SLURM resources."""
    return SlurmJob(
        stage,
        command,
        dependencies=dependencies,
        **resources_for(config, resource or stage, gpu),
    )


def _gate0_workflow(config: dict[str, Any]) -> list[SlurmJob]:
    """``build`` (array over splits) -> ``check`` (one job); the run stops here."""
    build = replace(_job(config, "build", "build", False), array_splits=(0, 1, 2))
    check = _job(config, "check", "check", False, (build.name,))
    return [build, check]


def _fit_group_job(
    config: dict[str, Any], group: str, dependencies: tuple[str, ...]
) -> SlurmJob:
    """One method group's sharded fit array (own resource key, own packing)."""
    shards_per_task = int(
        config.get("slurm", {}).get(
            f"fit_{group}_shards_per_task", _DEFAULT_SHARDS_PER_TASK[group]
        )
    )
    array_size = math.ceil(len(fit_units(group)) / shards_per_task)
    return replace(
        _job(
            config,
            f"fit-{group}",
            f"fit-shard --group {group} --shards-per-task {shards_per_task}",
            True,
            dependencies,
            resource="fit",
        ),
        array_size=array_size,
    )


def _fit_workflow(config: dict[str, Any]) -> list[SlurmJob]:
    """``import-anchor`` -> ``fit-standard``, ``fit-semantic-scale`` -> ``analyze``."""
    anchor = _job(config, "import-anchor", "import-anchor", False)
    standard = _fit_group_job(config, "standard", (anchor.name,))
    semantic_scale = _fit_group_job(config, "semantic_scale", (anchor.name,))
    analyze = _job(
        config, "analyze", "analyze", False, (standard.name, semantic_scale.name)
    )
    return [anchor, standard, semantic_scale, analyze]


def build_workflow(config: dict[str, Any], stage: str) -> list[SlurmJob]:
    """One phase of exp-3's DAG: ``'build'`` (Gate 0) or ``'fit'`` (post-approval)."""
    if stage == "build":
        return _gate0_workflow(config)
    if stage == "fit":
        return _fit_workflow(config)
    raise ValueError(f"Unknown submit stage: {stage!r}; expected 'build' or 'fit'")


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
    stage: str,
    config_path: str | None = None,
    dry_run: bool = False,
    submit: Callable[[str, bool], str] = _submit_script,
) -> dict[str, str]:
    """Render and submit one phase's jobs in dependency order.

    Copied from ``hydra/workflow.py::submit_workflow``'s submit loop (not
    imported: that function's ``build_workflow`` hard-codes exp-2's own DAG).
    ``check_queue_cap`` runs before every real submission; skipped in
    ``dry_run`` since no job is actually queued and ``squeue`` may not even
    be available off-cluster.
    """
    submitted: dict[str, str] = {}
    for job in build_workflow(config, stage):
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
