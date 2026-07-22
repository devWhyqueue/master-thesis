from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def run_shard_bundle(args: argparse.Namespace) -> bool:
    """Launch independent shard processes inside one accelerator allocation."""
    size = args.shards_per_task
    if size == 1:
        return False
    if size < 1:
        raise ValueError("shards-per-task must be positive")
    first = args.shard_index * size
    processes = [
        subprocess.Popen(_shard_command(args, index))
        for index in range(first, first + size)
    ]
    failures = [process.wait() for process in processes]
    if any(failures):
        raise RuntimeError(f"Shard bundle failed with exit codes {failures}")
    return True


def _shard_command(args: argparse.Namespace, shard_index: int) -> list[str]:
    entrypoint = Path(__file__).parents[4] / "__main__.py"
    command = [sys.executable, str(entrypoint)]
    if args.config:
        command.extend(("--config", args.config))
    command.extend(
        (
            "tune-shard",
            "--phase",
            args.phase,
            "--group",
            args.group,
            "--shard-index",
            str(shard_index),
        )
    )
    if args.observation_index is not None:
        command.extend(("--observation-index", str(args.observation_index)))
    if args.observations_per_candidate != 1:
        command.extend(
            ("--observations-per-candidate", str(args.observations_per_candidate))
        )
    if args.shard_offset:
        command.extend(("--shard-offset", str(args.shard_offset)))
    return command
