"""Command-line entry point for the coverage-similarity experiment."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Callable

import _bootstrap  # noqa: F401
from imbalance_benchmark.common import load_config

from breadth.slurm import _job, submit_workflow

from similarity.census import run_census
from similarity.precision import run_precision

__all__ = ["main"]


def _parser() -> argparse.ArgumentParser:
    """Create the coverage-similarity command-line argument parser."""
    parser = argparse.ArgumentParser(description="Coverage-Similarity Experiment CLI")
    parser.add_argument("--config", required=True)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("precision")
    sub.add_parser("census")

    submit = sub.add_parser("submit")
    submit.add_argument("--dry-run", action="store_true")

    return parser


def cmd_precision(args: argparse.Namespace) -> None:
    """Simulate the primary/breadth contrasts and select the fresh-draw count."""
    run_precision(load_config(args.config))


def cmd_census(args: argparse.Namespace) -> None:
    """Search all cohorts and run the manipulation check."""
    run_census(load_config(args.config))


def cmd_submit(args: argparse.Namespace) -> None:
    """Render and submit the experiment's SLURM workflow: precision -> census."""
    config = load_config(args.config)
    config_path = str(Path(args.config).resolve())
    precision = _job(config, "precision", "precision")
    census = _job(config, "census", "census", dependencies=(precision.name,))
    submit_workflow(config, config_path, args.dry_run, jobs=[precision, census])


def _commands() -> dict[str, Callable[[argparse.Namespace], None]]:
    """Return CLI dispatch table."""
    return {
        "precision": cmd_precision,
        "census": cmd_census,
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
