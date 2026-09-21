"""Command-line entry point for the prevalence imbalance experiment."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Callable

import _bootstrap  # noqa: F401
from imbalance_benchmark.common import load_config

from breadth.slurm import _job, submit_workflow

from centre.fit import shard_count

from prevalence.analyze import run_analyze
from prevalence.fit import run_fit_shard

__all__ = ["main"]

_STAGES = ("fit", "analyze", "all")


def _parser() -> argparse.ArgumentParser:
    """Create the prevalence imbalance command-line argument parser."""
    parser = argparse.ArgumentParser(description="Prevalence Imbalance Experiment CLI")
    parser.add_argument("--config", required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("fit")
    fit.add_argument("--shard-index", type=int, required=True)
    sub.add_parser("analyze")
    submit = sub.add_parser("submit")
    submit.add_argument("--stage", choices=_STAGES, required=True)
    submit.add_argument("--dry-run", action="store_true")
    return parser


def cmd_fit(args: argparse.Namespace) -> None:
    """Fit every pending prevalence arm of one (split, draw) shard."""
    run_fit_shard(load_config(args.config), args.shard_index)


def cmd_analyze(args: argparse.Namespace) -> None:
    """Pool arm accuracy, probability quality, and rank diagnostics; write the figure."""
    run_analyze(load_config(args.config))


def cmd_submit(args: argparse.Namespace) -> None:
    """Submit one stage, or fit -> analyze chained with afterok for ``all``."""
    config = load_config(args.config)
    chained = args.stage == "all"
    fit = _job(config, "fit", "fit", array_size=shard_count())
    analyze = _job(config, "analyze", "analyze", ("fit",) if chained else ())
    stages = {"fit": fit, "analyze": analyze}
    jobs = list(stages.values()) if chained else [stages[args.stage]]
    submit_workflow(config, str(Path(args.config).resolve()), args.dry_run, jobs=jobs)


def _commands() -> dict[str, Callable[[argparse.Namespace], None]]:
    """Return CLI dispatch table."""
    return {
        "fit": cmd_fit,
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
