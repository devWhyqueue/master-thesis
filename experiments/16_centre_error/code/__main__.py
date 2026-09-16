"""Command-line entry point for the centre-error experiment."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Callable

import _bootstrap  # noqa: F401
from imbalance_benchmark.common import load_config

from breadth.slurm import _job, submit_workflow

from centre import N_SPLITS
from centre.analyze import run_analyze
from centre.fit import run_fit_shard, shard_count
from centre.pool import run_pool

__all__ = ["main"]

_STAGES = ("pool", "fit", "analyze", "all")


def _parser() -> argparse.ArgumentParser:
    """Create the centre-error command-line argument parser."""
    parser = argparse.ArgumentParser(description="Centre Error Experiment CLI")
    parser.add_argument("--config", required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    pool = sub.add_parser("pool")
    pool.add_argument("--shard-index", type=int, choices=range(N_SPLITS), required=True)
    fit = sub.add_parser("fit")
    fit.add_argument("--shard-index", type=int, required=True)
    sub.add_parser("analyze")
    submit = sub.add_parser("submit")
    submit.add_argument("--stage", choices=_STAGES, required=True)
    submit.add_argument("--dry-run", action="store_true")
    return parser


def cmd_pool(args: argparse.Namespace) -> None:
    """Compute one split's pool centres, bases, and patient deviations."""
    run_pool(load_config(args.config), args.shard_index)


def cmd_fit(args: argparse.Namespace) -> None:
    """Fit every arm of one (split, draw) shard."""
    run_fit_shard(load_config(args.config), args.shard_index)


def cmd_analyze(args: argparse.Namespace) -> None:
    """Pool arm accuracy and write gaps, shares, and contrasts."""
    run_analyze(load_config(args.config))


def cmd_submit(args: argparse.Namespace) -> None:
    """Submit one stage, or pool -> fit -> analyze chained with afterok for ``all``."""
    config = load_config(args.config)
    chained = args.stage == "all"
    pool = _job(config, "pool", "pool", array_size=N_SPLITS)
    fit = _job(
        config, "fit", "fit", ("pool",) if chained else (), array_size=shard_count()
    )
    analyze = _job(config, "analyze", "analyze", ("fit",) if chained else ())
    stages = {"pool": pool, "fit": fit, "analyze": analyze}
    jobs = list(stages.values()) if chained else [stages[args.stage]]
    submit_workflow(config, str(Path(args.config).resolve()), args.dry_run, jobs=jobs)


def _commands() -> dict[str, Callable[[argparse.Namespace], None]]:
    """Return CLI dispatch table."""
    return {
        "pool": cmd_pool,
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
