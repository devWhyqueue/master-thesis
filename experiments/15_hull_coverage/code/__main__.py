"""Command-line entry point for the hull-coverage experiment."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Callable

import _bootstrap  # noqa: F401
from imbalance_benchmark.common import load_config

from breadth.slurm import _job, submit_workflow

from hull import N_SPLITS
from hull.census import run_census_shard
from hull.inference.precision import run_precision

__all__ = ["main"]


def _parser() -> argparse.ArgumentParser:
    """Create the hull-coverage command-line argument parser."""
    parser = argparse.ArgumentParser(description="Hull Coverage Experiment CLI")
    parser.add_argument("--config", required=True)
    sub = parser.add_subparsers(dest="command", required=True)

    census = sub.add_parser("census")
    census.add_argument(
        "--shard-index", type=int, choices=range(N_SPLITS), required=True
    )

    sub.add_parser("precision")

    submit = sub.add_parser("submit")
    submit.add_argument("--stage", choices=("census", "precision"), required=True)
    submit.add_argument("--dry-run", action="store_true")
    return parser


def cmd_census(args: argparse.Namespace) -> None:
    """Search one split's cohorts and run its manipulation check."""
    run_census_shard(load_config(args.config), args.shard_index)


def cmd_precision(args: argparse.Namespace) -> None:
    """Simulate the class-recall model and select the draw count."""
    run_precision(load_config(args.config))


def cmd_submit(args: argparse.Namespace) -> None:
    """Submit one gated stage: the census array, or the precision simulation after the check passes."""
    config = load_config(args.config)
    array_size = N_SPLITS if args.stage == "census" else 0
    job = _job(config, args.stage, args.stage, array_size=array_size)
    submit_workflow(config, str(Path(args.config).resolve()), args.dry_run, jobs=[job])


def _commands() -> dict[str, Callable[[argparse.Namespace], None]]:
    """Return CLI dispatch table."""
    return {"census": cmd_census, "precision": cmd_precision, "submit": cmd_submit}


def main() -> None:
    """Parse command-line arguments and dispatch subcommand."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = _parser().parse_args()
    _commands()[args.command](args)


if __name__ == "__main__":
    main()
