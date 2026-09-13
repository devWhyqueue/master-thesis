from __future__ import annotations

from dataclasses import replace
from typing import Any

from imbalance_benchmark.hydra.job_resources import build_job
from imbalance_benchmark.hydra.rendering import SlurmJob
from imbalance_benchmark.modeling.workflows.confirmation_schedule import (
    confirm_array_size,
)

__all__ = ["confirm_jobs"]


def _confirm_group_job(
    config: dict[str, Any], group: str, is_mil: bool, shards_per_task: int
) -> SlurmJob:
    """Build one confirm group's sharded array job (its own resource key/partition)."""
    return replace(
        build_job(
            config,
            f"confirm-{group}",
            f"confirm-shard --group {group} --shards-per-task {shards_per_task}",
            True,
            (),
            f"confirm_{group}",
            "confirm",
        ),
        array_size=confirm_array_size(group, is_mil, shards_per_task),
    )


def confirm_jobs(config: dict[str, Any]) -> tuple[SlurmJob, SlurmJob]:
    """Shard confirmation into a natural (gpu-5h) and controlled (gpu-2h) array.

    Mirrors the tuning stage's natural/controlled resource split: natural
    fits are long but few, controlled fits are cheap but many, so each group
    gets its own bundle-size knob and partition instead of sharing one
    two-day allocation across both. Every array task resolves its own unit
    bundle from ``confirmation_schedule`` at run time; no reduce step is
    needed because ``analyze`` discovers run records by directory glob.
    Submitted separately (``submit --confirm-only``) once every condition's
    tuning lock is resolved - see ``build_workflow``'s ``confirm_only``.
    """
    is_mil = config.get("dataset", {}).get("regime", "patch") == "wsi"
    sl = config.get("slurm", {})
    natural = _confirm_group_job(
        config, "natural", is_mil, int(sl.get("confirm_natural_shards_per_task", 1))
    )
    controlled = _confirm_group_job(
        config,
        "controlled",
        is_mil,
        int(sl.get("confirm_controlled_shards_per_task", 1)),
    )
    return natural, controlled
