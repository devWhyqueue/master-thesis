from __future__ import annotations

from dataclasses import replace
from typing import Any

from imbalance_benchmark.hydra.job_resources import build_job as _job
from imbalance_benchmark.hydra.rendering import SlurmJob
from imbalance_benchmark.modeling.workflows.tuning.tuning_schedule import (
    bundled_array_size,
    candidate_array_size,
    phase_methods,
)

__all__ = ["dependent_round_zero_jobs", "final_reduce_job"]


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
    """Build the dependent phase's frozen round-0 training jobs (crt, post-hoc).

    Shaped exactly like the pre-adaptive-search static jobs, but submitted
    by ``tune-decide`` once CE specifically resolves, rather than
    statically after base's round-0 reduce - CE's own adaptive search may
    still need further rounds at that point. Only the controlled conditions
    appear: the natural anchor fits CE alone, so it has no dependent method.
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
