"""Command-line entry point for the prior-injection margin experiment (exp-33)."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Callable

import _bootstrap  # noqa: F401
from imbalance_benchmark.common import load_config

from breadth.slurm import _job, submit_workflow

from margin.analyze import run_analyze
from margin.inject import run_inject

__all__ = ["main"]

_STAGES = ("inject", "analyze", "all")


def _parser() -> argparse.ArgumentParser:
    """Create the margin command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Prior-Injection Margin Experiment CLI"
    )
    parser.add_argument("--config", required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("inject")
    sub.add_parser("analyze")
    submit = sub.add_parser("submit")
    submit.add_argument("--stage", choices=_STAGES, required=True)
    submit.add_argument("--dry-run", action="store_true")
    return parser


def cmd_inject(args: argparse.Namespace) -> None:
    """Write every Q{rho}/QT{rho} run record for this dataset (0 new fits)."""
    run_inject(load_config(args.config))


def cmd_analyze(args: argparse.Namespace) -> None:
    """Run the gate, H1-H3, and write analysis, diagnostics, and figures."""
    run_analyze(load_config(args.config))


def cmd_submit(args: argparse.Namespace) -> None:
    """Submit one stage, or inject -> analyze chained with afterok for ``all``."""
    config = load_config(args.config)
    chained = args.stage == "all"
    inject = _job(config, "inject", "inject")
    analyze = _job(config, "analyze", "analyze", ("inject",) if chained else ())
    stages = {"inject": inject, "analyze": analyze}
    jobs = list(stages.values()) if chained else [stages[args.stage]]
    submit_workflow(config, str(Path(args.config).resolve()), args.dry_run, jobs=jobs)


def _commands() -> dict[str, Callable[[argparse.Namespace], None]]:
    """Return CLI dispatch table."""
    return {
        "inject": cmd_inject,
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
