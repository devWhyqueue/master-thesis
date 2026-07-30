from __future__ import annotations

from dataclasses import replace
from typing import Any

from imbalance_benchmark.hydra.job_resources import build_job as _job
from imbalance_benchmark.hydra.rendering import SlurmJob
from imbalance_benchmark.modeling.context import roster_for_regime
from imbalance_benchmark.modeling.workflows.tuning.tuning_schedule import (
    DEPENDENT_METHODS,
    bundled_array_size,
    bundled_observation_array_size,
    candidate_array_size,
)

__all__ = ["dependent_round_zero_jobs"]


def _posthoc_natural_job(
    config: dict[str, Any], dependency: tuple[str, ...]
) -> SlurmJob:
    return _job(
        config,
        "tune-dependent-posthoc-natural",
        "tune-shard --phase dependent --group natural --shard-index 0",
        True,
        dependency,
        "tune_post_hoc_natural",
        "tune_natural",
    )


def _crt_natural_job(
    config: dict[str, Any],
    dependency: tuple[str, ...],
    natural_observations: int,
    bundle_arg: str,
) -> SlurmJob:
    shards_per_task = int(config.get("slurm", {}).get("tune_shards_per_task", 1))
    return replace(
        _job(
            config,
            "tune-dependent-crt-natural",
            "tune-shard --phase dependent --group natural"
            f" --observations-per-candidate {natural_observations}"
            f" --shard-offset 1 --bundle-by-observation{bundle_arg}",
            True,
            dependency,
            "tune_natural",
            "tune",
        ),
        array_size=bundled_observation_array_size(
            candidate_array_size(("crt",)), natural_observations, shards_per_task
        ),
    )


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
            3 * candidate_array_size(dependent_methods), shards_per_task
        ),
    )


def dependent_round_zero_jobs(
    config: dict[str, Any], is_mil: bool, dependency: tuple[str, ...] = ()
) -> list[SlurmJob]:
    """Build the dependent phase's frozen round-0 training jobs (crt, post-hoc).

    Shaped exactly like the pre-adaptive-search static jobs, but submitted
    by ``tune-decide`` once CE specifically resolves, rather than
    statically after base's round-0 reduce - CE's own adaptive search may
    still need further rounds at that point.
    """
    roster = roster_for_regime(is_mil)
    dependent_methods = tuple(
        method for method in roster if method in DEPENDENT_METHODS
    )
    natural_observations = int(
        config.get("slurm", {}).get("tune_natural_observations_per_candidate", 1)
    )
    shards_per_task = int(config.get("slurm", {}).get("tune_shards_per_task", 1))
    bundle_arg = f" --shards-per-task {shards_per_task}"
    return [
        _posthoc_natural_job(config, dependency),
        _crt_natural_job(config, dependency, natural_observations, bundle_arg),
        _controlled_job(config, dependency, dependent_methods, bundle_arg),
    ]
