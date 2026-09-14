"""Command-line entry point for the coverage-redundancy experiment."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Callable

import _bootstrap  # noqa: F401
from imbalance_benchmark.common import load_config

from breadth.slurm import _job, submit_workflow

from coverage_redundancy.analyze import run_analyze
from coverage_redundancy.census import run_census


def _parser() -> argparse.ArgumentParser:
    """Create the coverage-redundancy command-line argument parser."""
    parser = argparse.ArgumentParser(description="Coverage-Redundancy Experiment CLI")
    parser.add_argument("--config", required=True)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("census")
    sub.add_parser("analyze")

    submit = sub.add_parser("submit")
    submit.add_argument("--dry-run", action="store_true")

    return parser


def cmd_census(args: argparse.Namespace) -> None:
    """Census every random and composition cohort's tau^2, omega, Neff^omega, r."""
    run_census(load_config(args.config))


def cmd_analyze(args: argparse.Namespace) -> None:
    """Fit the residual breadth surface, predict cohort composition, and label it."""
    run_analyze(load_config(args.config))


def cmd_submit(args: argparse.Namespace) -> None:
    """Render and submit the experiment's SLURM workflow: census -> analyze."""
    config = load_config(args.config)
    config_path = str(Path(args.config).resolve())
    census = _job(config, "census", "census")
    analyze = _job(config, "analyze", "analyze", dependencies=(census.name,))
    submit_workflow(config, config_path, args.dry_run, jobs=[census, analyze])


def _commands() -> dict[str, Callable[[argparse.Namespace], None]]:
    """Return CLI dispatch table."""
    return {
        "census": cmd_census,
        "analyze": cmd_analyze,
        "submit": cmd_submit,
    }


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
