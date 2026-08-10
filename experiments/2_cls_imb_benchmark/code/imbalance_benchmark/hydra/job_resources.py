from __future__ import annotations

from dataclasses import replace
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


def stage_jobs(
    config: dict[str, Any], stage: str, split_index: int | None
) -> list[SlurmJob]:
    """Build one selected PANDA readiness boundary without later work."""
    if config.get("dataset", {}).get("name") != "panda":
        raise ValueError("Stage-only submission is reserved for PANDA readiness")
    if stage == "materialize":
        return _materialize_stage_jobs(config)
    if stage == "extract":
        return _extract_stage_jobs(config)
    if stage == "prepare":
        return [build_job(config, "prepare", "prepare", False)]
    if stage in {"pilot", "freeze"}:
        splits = (split_index,) if split_index is not None else (0, 1, 2)
        return [replace(build_job(config, stage, stage, False), array_splits=splits)]
    raise ValueError(f"Unknown stage-only boundary: {stage}")


def _materialize_stage_jobs(config: dict[str, Any]) -> list[SlurmJob]:
    """Build the audit array -> combine -> pack array -> publish dependency chain."""
    mp = config.get("materialize_panda", {})
    audit_count = int(mp.get("audit_shard_count", 32))
    pack_count = int(mp.get("shard_count", 48))
    audit = replace(
        build_job(config, "materialize_audit", "materialize-panda-audit", False),
        array_size=audit_count,
    )
    combine = build_job(
        config, "materialize_combine", "materialize-panda-combine", False, (audit.name,)
    )
    pack = replace(
        build_job(
            config, "materialize_pack", "materialize-panda-pack", False, (combine.name,)
        ),
        array_size=pack_count,
    )
    publish = build_job(
        config, "materialize_publish", "materialize-panda-publish", False, (pack.name,)
    )
    return [audit, combine, pack, publish]


def _extract_stage_jobs(config: dict[str, Any]) -> list[SlurmJob]:
    count = int(config.get("materialize_panda", {}).get("shard_count", 48))
    extract = replace(
        build_job(
            config,
            "prepare-extract-shard",
            f"prepare-extract-shard --shard-count {count}",
            True,
            resource="extract",
        ),
        array_size=count,
    )
    reduce = build_job(
        config,
        "prepare-extract-reduce",
        "prepare-extract-reduce",
        False,
        (extract.name,),
        "extract_reduce",
    )
    return [extract, reduce]
