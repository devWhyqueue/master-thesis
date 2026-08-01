from __future__ import annotations

from typing import Any

from imbalance_benchmark.hydra.rendering import SlurmJob

__all__ = ["resources_for", "build_job"]


def resources_for(
    config: dict[str, Any], stage: str, gpu: bool, fallback: str | None = None
) -> dict[str, Any]:
    """Resolve one stage's SLURM resources, falling back to a shared resource key."""
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
        "constraint": str(sr["constraint"]) if sr.get("constraint") else None,
    }


def build_job(
    config: dict[str, Any],
    stage: str,
    cmd: str,
    gpu: bool,
    dependencies: tuple[str, ...] = (),
    resource: str | None = None,
    fallback: str | None = None,
) -> SlurmJob:
    """Build one stage's job with its resolved SLURM resources.

    ``tune-decide`` shells out to ``sbatch``/``squeue`` itself to self-chain
    the adaptive search - the Apptainer container has no SLURM client, so
    that resource type always runs on the host instead.
    """
    return SlurmJob(
        stage,
        cmd,
        dependencies=dependencies,
        on_host=resource == "tune_decide",
        **resources_for(config, resource or stage, gpu, fallback),
    )
