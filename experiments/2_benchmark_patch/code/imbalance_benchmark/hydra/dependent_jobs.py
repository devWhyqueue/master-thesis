from __future__ import annotations

from dataclasses import replace
from typing import Any

from imbalance_benchmark.hydra.job_resources import build_job as _job
from imbalance_benchmark.hydra.rendering import SlurmJob
from imbalance_benchmark.hydra.resume import ResumePlan
from imbalance_benchmark.modeling.context import CONTROLLED_CONDITIONS
from imbalance_benchmark.modeling.workflows.tuning.tuning_schedule import (
    bundled_array_size,
    bundled_observation_array_size,
    candidate_array_size,
    phase_methods,
)

__all__ = ["base_tuning_jobs", "dependent_round_zero_jobs", "final_reduce_job"]


def _controlled_job(
    config: dict[str, Any],
    dependency: tuple[str, ...],
    dependent_methods: tuple[str, ...],
    bundle_arg: str,
) -> SlurmJob:
    shards_per_task = int(config.get("slurm", {}).get("tune_shards_per_task", 1))
    return replace(
        _job(
            config,
            "tune-dependent-controlled",
            f"tune-shard --phase dependent --group controlled{bundle_arg}",
            True,
            dependency,
            "tune_controlled",
            "tune",
        ),
        array_size=bundled_array_size(
            len(CONTROLLED_CONDITIONS) * candidate_array_size(dependent_methods),
            shards_per_task,
        ),
    )


def final_reduce_job(config: dict[str, Any], condition: str) -> SlurmJob:
    """Sign one condition's final tuning selection once its dependent phase converges."""
    return _job(
        config,
        f"tune-final-reduce-{condition}",
        f"tune-reduce --phase final --condition {condition}",
        False,
        (),
        "tune_reduce",
    )


def dependent_round_zero_jobs(
    config: dict[str, Any], is_mil: bool, dependency: tuple[str, ...] = ()
) -> list[SlurmJob]:
    """Build the CE-inherited search's frozen round-0 training jobs (crt, post-hoc).

    Shaped exactly like the pre-adaptive-search static jobs, but submitted
    by ``tune-decide`` once a condition's CE config specifically resolves,
    rather than statically after the self-contained search's round-0 reduce
    - CE's own adaptive search may still need further rounds at that point.
    Only the controlled conditions appear: the natural anchor fits CE alone,
    so it has no CE-inherited method.
    """
    controlled_shards = int(config.get("slurm", {}).get("tune_shards_per_task", 1))
    return [
        _controlled_job(
            config,
            dependency,
            phase_methods(is_mil, "dependent", "balanced"),
            f" --shards-per-task {controlled_shards}",
        )
    ]


def _packed_job(
    config: dict[str, Any],
    name: str,
    command: str,
    resource: str,
    freeze_dependency: tuple[str, ...],
    parallel_fits: int,
    array_size: int,
    array_indices: tuple[int, ...],
) -> SlurmJob:
    """Build one base-phase tune-shard array job, scaling cpus by packed fits."""
    job = _job(config, name, command, True, freeze_dependency, resource, "tune")
    return replace(
        job,
        array_size=array_size,
        array_indices=array_indices,
        cpus=job.cpus * parallel_fits,
    )


def _base_natural_job(
    config: dict[str, Any],
    slurm: dict[str, Any],
    freeze_dependency: tuple[str, ...],
    plan: ResumePlan | None,
    parallel_fits: int,
) -> SlurmJob | None:
    if plan and not plan.natural_indices:
        return None
    is_mil = config.get("dataset", {}).get("regime", "patch") == "wsi"
    observations = int(slurm.get("tune_natural_observations_per_candidate", 1))
    shards = int(
        slurm.get("tune_natural_shards_per_task", slurm.get("tune_shards_per_task", 1))
    )
    methods = phase_methods(is_mil, "base", "natural")
    return _packed_job(
        config,
        "tune-base-natural",
        "tune-shard --phase base --group natural"
        f" --observations-per-candidate {observations}"
        f" --bundle-by-observation --shards-per-task {shards}"
        f" --parallel-fits {parallel_fits}",
        "tune_natural",
        freeze_dependency,
        parallel_fits,
        bundled_observation_array_size(
            candidate_array_size(methods), observations, shards
        ),
        plan.natural_indices if plan else (),
    )


def _base_controlled_job(
    config: dict[str, Any],
    slurm: dict[str, Any],
    freeze_dependency: tuple[str, ...],
    plan: ResumePlan | None,
    parallel_fits: int,
) -> SlurmJob | None:
    if plan and not plan.controlled_indices:
        return None
    is_mil = config.get("dataset", {}).get("regime", "patch") == "wsi"
    shards = int(slurm.get("tune_shards_per_task", 1))
    methods = phase_methods(is_mil, "base", "balanced")
    return _packed_job(
        config,
        "tune-base-controlled",
        "tune-shard --phase base --group controlled"
        f" --shards-per-task {shards}"
        f" --parallel-fits {parallel_fits}",
        "tune_controlled",
        freeze_dependency,
        parallel_fits,
        bundled_array_size(
            len(CONTROLLED_CONDITIONS) * candidate_array_size(methods), shards
        ),
        plan.controlled_indices if plan else (),
    )


def base_tuning_jobs(
    config: dict[str, Any], freeze_dependency: tuple[str, ...], plan: ResumePlan | None
) -> tuple[SlurmJob | None, SlurmJob | None]:
    """Build round-0 base tuning's natural and controlled array jobs, packed per GPU."""
    slurm = config.get("slurm", {})
    parallel_fits = int(slurm.get("tune_parallel_fits", 1))
    return (
        _base_natural_job(config, slurm, freeze_dependency, plan, parallel_fits),
        _base_controlled_job(config, slurm, freeze_dependency, plan, parallel_fits),
    )
