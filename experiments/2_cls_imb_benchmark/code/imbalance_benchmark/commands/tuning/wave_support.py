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


def next_stalled(data: Path, pending: list[int], current: int) -> int:
    """Persist artifact progress and stop after three unchanged rescans."""
    state_path = data / "tuning_wave_state.json"
    remaining = len(pending)
    prior = json.loads(state_path.read_text()) if state_path.exists() else {}
    stalled = current + 1 if prior.get("remaining") == remaining else 0
    if stalled >= 3:
        raise RuntimeError(
            "Three tuning waves produced no artifacts; stopping. Resume with: submit --resume-tuning"
        )
    state_path.write_text(json.dumps({"remaining": remaining}) + "\n")
    return stalled


def submit_wave(
    config: dict[str, Any],
    config_path: str,
    jobs: list[SlurmJob],
    args: argparse.Namespace,
    stalled: int,
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
        f"{command} --stalled-waves {stalled}",
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
