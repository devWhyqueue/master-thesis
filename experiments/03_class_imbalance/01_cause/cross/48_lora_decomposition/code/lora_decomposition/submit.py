"""Sparse one-arm-per-task SLURM arrays for the pilot/main phases, capped by the account queue.

Pending work is every (arm, shard) of the phase without a compatible completed run and not
already queued by an earlier submission of this experiment (``submissions.json`` keeps each array
id with its arm and shard indices). Every prospective array worker counts against the 100-task
account limit: an array is trimmed to the free slots, the queue is recounted before each further
array, and whatever does not fit stays pending for a later invocation.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from imbalance_benchmark.common import output_root
from imbalance_benchmark.hydra.guards import DEFAULT_QUEUE_CAP
from imbalance_benchmark.hydra.job_resources import build_job
from imbalance_benchmark.hydra.rendering import SlurmJob, render_sbatch

from centre.fit import decode_shard_index, shard_count

from centre import N_DRAWS, N_SPLITS

from lora_decomposition import protocol
from lora_decomposition.fit import is_marked
from lora_decomposition.protocol import ARMS, is_smoke, validate_protocol

__all__ = [
    "check_indices",
    "phase_indices",
    "record_submission",
    "submissions_path",
    "submit_analyze",
    "submit_jobs",
]

logger = logging.getLogger(__name__)

_PILOT = tuple(s * N_DRAWS for s in range(N_SPLITS))


def check_indices(indices: list[int]) -> None:
    """Raise on any shard index outside the 30 (split, draw) blocks."""
    bad = [i for i in indices if i not in range(shard_count())]
    if bad:
        raise ValueError(f"shard indices out of range: {bad}")


def phase_indices(phase: str) -> tuple[int, ...]:
    """Pilot: draw 0 of every split; main: the other 27 blocks; smoke: block 0 only."""
    if phase == "pilot":
        return _PILOT
    if phase == "main":
        return tuple(i for i in range(shard_count()) if i not in _PILOT)
    if phase == "smoke":
        return (0,)
    raise ValueError(f"unknown phase {phase!r}")


def submissions_path(config: dict[str, Any]) -> Path:
    """Log of every submitted array: job id, arm, shard indices, phase."""
    return output_root(config) / "submissions.json"


def _submissions(config: dict[str, Any]) -> list[dict[str, Any]]:
    path = submissions_path(config)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []


def record_submission(
    config: dict[str, Any], job_id: str, arm: str, indices: list[int], phase: str
) -> None:
    """Append one submitted array to the experiment's submission log."""
    entries = _submissions(config)
    entries.append(
        {
            "job_id": job_id,
            "arm": arm,
            "indices": indices,
            "phase": phase,
            "at": time.time(),
        }
    )
    path = submissions_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, indent=2), encoding="utf-8")


def _squeue_ids() -> list[str]:
    """This account's queued/running task ids, one array element per line."""
    user = os.environ.get("USER") or os.environ.get("USERNAME", "")
    result = subprocess.run(
        ["squeue", "-u", user, "-r", "-h", "-o", "%i"],
        capture_output=True,
        text=True,
        check=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _sbatch(script: str) -> str:
    result = subprocess.run(
        ["sbatch", "--parsable"],
        input=script,
        text=True,
        check=True,
        capture_output=True,
    )
    return result.stdout.strip().split(";", maxsplit=1)[0]


def _in_flight(config: dict[str, Any], queued: list[str]) -> set[tuple[str, int]]:
    """(arm, shard) pairs of this experiment's earlier arrays still in the queue."""
    ids = set(queued)
    return {
        (entry["arm"], i)
        for entry in _submissions(config)
        for i in entry["indices"]
        if f"{entry['job_id']}_{i}" in ids
        or any(q.startswith(f"{entry['job_id']}_[") for q in ids)
    }


def _require_pilot_review() -> None:
    report = protocol.EXPERIMENT_DIR / "report"
    check_file, review = report / "check_pilot.json", report / "pilot_check.md"
    passed = check_file.is_file() and json.loads(
        check_file.read_text(encoding="utf-8")
    ).get("passed")
    if not (passed and review.is_file()):
        raise ValueError(
            "main phase needs a passed pilot check and report/pilot_check.md"
        )


def _pending(
    config: dict[str, Any], phase: str, in_flight: set[tuple[str, int]]
) -> dict[str, list[int]]:
    """Per arm, the phase's shard indices with neither a compatible run nor a queued task."""
    return {
        arm: [
            i
            for i in phase_indices(phase)
            if (arm, i) not in in_flight
            and not is_marked(config, *decode_shard_index(i), arm)
        ]
        for arm in (("B",) if phase == "smoke" else ARMS)
    }


def _fit_job(config: dict[str, Any], arm: str, indices: list[int]) -> SlurmJob:
    """One arm's sparse array; job name stays ``fit`` so the dataset SquashFS mounts apply."""
    check_indices(indices)
    return replace(
        build_job(config, "fit", f"fit --arm {arm}", True, resource="fit"),
        array_size=shard_count(),
        array_indices=tuple(indices),
    )


def _check_phase(config: dict[str, Any], phase: str) -> None:
    validate_protocol(config)
    if (phase == "smoke") != is_smoke(config):
        raise ValueError(
            "the smoke phase needs a smoke config, and only the smoke phase"
        )
    if phase == "main":
        _require_pilot_review()


def submit_jobs(
    config: dict[str, Any],
    phase: str,
    dry_run: bool,
    *,
    queue: Callable[[], list[str]] = _squeue_ids,
    sbatch: Callable[[str], str] = _sbatch,
) -> list[str]:
    """Submit the phase's pending work as one sparse array per arm, within free queue slots."""
    _check_phase(config, phase)
    in_flight = set() if dry_run else _in_flight(config, queue())
    submitted: list[str] = []
    for arm, indices in _pending(config, phase, in_flight).items():
        if not dry_run:
            indices = indices[: max(DEFAULT_QUEUE_CAP - len(queue()), 0)]
        if not indices:
            continue
        script = render_sbatch(
            _fit_job(config, arm, indices), config, config["config_path"]
        )
        if dry_run:
            logger.info("dry run, arm %s shards %s:\n%s", arm, indices, script)
            submitted.append(f"dry-run-{arm}")
            continue
        job_id = sbatch(script)
        record_submission(config, job_id, arm, indices, phase)
        logger.info("arm %s shards %s: job %s", arm, indices, job_id)
        submitted.append(job_id)
    return submitted


def submit_analyze(
    configs: list[dict[str, Any]], dry_run: bool, sbatch: Callable[[str], str] = _sbatch
) -> str:
    """One CPU job analyzing both datasets, on the TCGA-UT config's analyze resources."""
    by_name = {c["dataset"]["name"]: c for c in configs}
    host, other = by_name["tcga_ut"], by_name["bracs"]
    job = build_job(host, "analyze", f"--config {other['config_path']} analyze", False)
    script = render_sbatch(job, host, host["config_path"])
    if dry_run:
        logger.info("dry run, analyze:\n%s", script)
        return "dry-run-analyze"
    return sbatch(script)
