"""Command-line entry point for the unified imbalance benchmark."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable

from imbalance_benchmark.commands import (
    cmd_analyze,
    cmd_combine_rq3,
    cmd_confirm,
    cmd_confirm_shard,
    cmd_freeze,
    cmd_pilot,
    cmd_prepare,
    cmd_smoke,
    cmd_submit,
    cmd_tune,
    cmd_tune_reduce,
    cmd_tune_shard,
)


def _parser() -> argparse.ArgumentParser:
    """Create the benchmark command-line parser."""
    parser = argparse.ArgumentParser(description="Class-Imbalance Benchmark CLI")
    parser.add_argument("--config", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-index", type=int, choices=range(3), default=None)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in (
        "prepare",
        "pilot",
        "freeze",
        "tune",
        "tune-shard",
        "tune-reduce",
        "confirm",
        "confirm-shard",
        "analyze",
        "combine-rq3",
        "smoke",
    ):
        sub.add_parser(command)
    submit = sub.add_parser("submit")
    submit.add_argument("--dry-run", action="store_true")
    submit.add_argument("--smoke", action="store_true")
    submit.add_argument("--resume-tuning", action="store_true")
    for command in ("tune", "confirm"):
        sub.choices[command].add_argument(
            "--condition", choices=("natural", "balanced", "moderate", "severe")
        )
    shard = sub.choices["tune-shard"]
    shard.add_argument("--phase", choices=("base", "dependent"), required=True)
    shard.add_argument("--group", choices=("natural", "controlled"), required=True)
    shard.add_argument("--shard-index", type=int, required=True)
    shard.add_argument("--observation-index", type=int)
    shard.add_argument("--observations-per-candidate", type=int, default=1)
    shard.add_argument("--shard-offset", type=int, default=0)
    shard.add_argument("--shards-per-task", type=int, default=1)
    shard.add_argument("--bundle-by-observation", action="store_true")
    reduce = sub.choices["tune-reduce"]
    reduce.add_argument("--phase", choices=("base", "final"), required=True)
    confirm_shard = sub.choices["confirm-shard"]
    confirm_shard.add_argument(
        "--group", choices=("natural", "controlled"), required=True
    )
    confirm_shard.add_argument("--shard-index", type=int, required=True)
    confirm_shard.add_argument("--shards-per-task", type=int, default=1)
    return parser


def _commands() -> dict[str, Callable[[argparse.Namespace], None]]:
    """Return CLI command handlers keyed by their subcommand names."""
    return {
        "prepare": cmd_prepare,
        "pilot": cmd_pilot,
        "freeze": cmd_freeze,
        "tune": cmd_tune,
        "tune-shard": cmd_tune_shard,
        "tune-reduce": cmd_tune_reduce,
        "confirm": cmd_confirm,
        "confirm-shard": cmd_confirm_shard,
        "analyze": cmd_analyze,
        "combine-rq3": cmd_combine_rq3,
        "submit": cmd_submit,
        "smoke": cmd_smoke,
    }


def main() -> None:
    """Parse benchmark command-line arguments and dispatch the selected command."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    args = _parser().parse_args()
    _commands()[args.command](args)


if __name__ == "__main__":
    main()
