"""Shared state and submission primitives for artifact-driven tuning waves."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Any

from imbalance_benchmark.hydra.job_resources import build_job
from imbalance_benchmark.hydra.rendering import SlurmJob, render_sbatch
from imbalance_benchmark.hydra.workflow import _submit_script


def _attempted_path(data: Path, scope: str) -> Path:
    return data / f"tuning_wave_attempted_{scope}.json"


def unattempted(data: Path, scope: str, pending: list[int]) -> list[int]:
    """Return pending indices no prior wave has already submitted.

    A shard that ran and failed (e.g. timed out) leaves no artifact, so it
    would otherwise look identical to one never submitted and get retried
    forever. Once an index has been attempted, it is never selected again;
    a stage with attempted-but-still-missing shards must be fixed and
    resumed explicitly rather than retried automatically.
    """
    path = _attempted_path(data, scope)
    tried = set(json.loads(path.read_text())) if path.exists() else set()
    return [index for index in pending if index not in tried]


def record_attempted(data: Path, scope: str, indices: list[int]) -> None:
    """Persist indices selected for submission so later waves skip them."""
    path = _attempted_path(data, scope)
    tried = set(json.loads(path.read_text())) if path.exists() else set()
    path.write_text(json.dumps(sorted(tried | set(indices))) + "\n")


def submit_wave(
    config: dict[str, Any],
    config_path: str,
    jobs: list[SlurmJob],
    args: argparse.Namespace,
) -> None:
    """Submit sparse arrays and exactly one afterany self-rescanning successor."""
    ids = [
        _submit_script(render_sbatch(job, config, config_path), False) for job in jobs
    ]
    command = f"tune-wave --phase {args.phase} --round {args.round}"
    if args.condition is not None:
        command += f" --condition {args.condition}"
    if getattr(args, "group", None) is not None:
        command += f" --group {args.group}"
    successor = build_job(
        config,
        "tune-wave",
        command,
        False,
        tuple(ids),
        "tune_decide",
        "tune_reduce",
    )
    _submit_script(
        render_sbatch(
            replace(successor, dependency_mode="afterany"), config, config_path
        ),
        False,
    )
