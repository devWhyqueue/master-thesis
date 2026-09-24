"""Command-line entry point for experiment 37: support coverage."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Callable

import _bootstrap  # noqa: F401
from imbalance_benchmark.common import load_config

from breadth.slurm import _job, submit_workflow

from support.analyze import run_analyze
from support.fit import run_fit_shard, shard_count
from support.precheck import run_precheck

__all__ = ["main"]

_STAGES = ("precheck", "fit", "analyze", "all")


def _parser() -> argparse.ArgumentParser:
    """Create the support-coverage command-line argument parser."""
    parser = argparse.ArgumentParser(description="Support Coverage Experiment CLI")
    parser.add_argument("--config", required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("precheck")
    fit = sub.add_parser("fit")
    fit.add_argument("--shard-index", type=int, required=True)
    sub.add_parser("analyze")
    submit = sub.add_parser("submit")
    submit.add_argument("--stage", choices=_STAGES, required=True)
    submit.add_argument("--dry-run", action="store_true")
    return parser


def cmd_precheck(args: argparse.Namespace) -> None:
    """Run gates 0-2: zero-fit geometry and (BRACS) the Step 1 fixed-lambda check."""
    run_precheck(load_config(args.config))


def cmd_fit(args: argparse.Namespace) -> None:
    """Fit B and every S arm of one (split, draw) shard, plus their fixed-lambda controls."""
    run_fit_shard(load_config(args.config), args.shard_index)


def cmd_analyze(args: argparse.Namespace) -> None:
    """Pool B/S-arm accuracy, compute D_S/rescue and any primary estimates/label."""
    run_analyze(load_config(args.config))


def _build_stages(config: dict, chained: bool) -> dict:
    """Build every stage's job, chained with afterok dependencies when ``chained``."""
    precheck = _job(config, "precheck", "precheck")
    fit = _job(
        config,
        "fit",
        "fit",
        (precheck.name,) if chained else (),
        array_size=shard_count(),
    )
    analyze = _job(config, "analyze", "analyze", (fit.name,) if chained else ())
    return {"precheck": precheck, "fit": fit, "analyze": analyze}


def cmd_submit(args: argparse.Namespace) -> None:
    """Submit one stage, or precheck -> fit -> analyze for ``all``."""
    config = load_config(args.config)
    chained = args.stage == "all"
    stages = _build_stages(config, chained)
    jobs = list(stages.values()) if chained else [stages[args.stage]]
    submit_workflow(config, str(Path(args.config).resolve()), args.dry_run, jobs=jobs)


def _commands() -> dict[str, Callable[[argparse.Namespace], None]]:
    """Return CLI dispatch table."""
    return {
        "precheck": cmd_precheck,
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
