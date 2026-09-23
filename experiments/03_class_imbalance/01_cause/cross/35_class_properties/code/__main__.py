"""Command-line entry point for the class-properties cross-dataset experiment (exp-35)."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Callable

import _bootstrap  # noqa: F401
from imbalance_benchmark.common import load_config

from breadth.slurm import _job, submit_workflow

from classprops.analyze import run_analyze
from classprops.extract import run_extract
from classprops.gate import run_gate, write_gate

__all__ = ["main"]

_STAGES = ("gate", "extract", "analyze", "all")


def _parser() -> argparse.ArgumentParser:
    """Create the classprops command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Class-Properties Cross-Dataset Experiment CLI"
    )
    parser.add_argument("--config", required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("gate")
    sub.add_parser("extract")
    sub.add_parser("analyze")
    submit = sub.add_parser("submit")
    submit.add_argument("--stage", choices=_STAGES, required=True)
    submit.add_argument("--dry-run", action="store_true")
    return parser


def cmd_gate(args: argparse.Namespace) -> None:
    """Run the pre-registered (headroom, margin) overlap manipulation check (0 new fits)."""
    config = load_config(args.config)
    write_gate(config, run_gate(config))


def cmd_extract(args: argparse.Namespace) -> None:
    """Build every pool's observations and this dataset's cross-fitted covariates (0 new fits)."""
    run_extract(load_config(args.config))


def cmd_analyze(args: argparse.Namespace) -> None:
    """Fit M0/M1 across both datasets' pools and write the shrink share and label."""
    run_analyze(load_config(args.config))


def cmd_submit(args: argparse.Namespace) -> None:
    """Submit one stage, or extract -> analyze chained with afterok for ``all``."""
    config = load_config(args.config)
    chained = args.stage == "all"
    gate = _job(config, "gate", "gate")
    extract = _job(config, "extract", "extract")
    analyze = _job(config, "analyze", "analyze", ("extract",) if chained else ())
    stages = {"gate": gate, "extract": extract, "analyze": analyze}
    jobs = [stages["extract"], stages["analyze"]] if chained else [stages[args.stage]]
    submit_workflow(config, str(Path(args.config).resolve()), args.dry_run, jobs=jobs)


def _commands() -> dict[str, Callable[[argparse.Namespace], None]]:
    """Return CLI dispatch table."""
    return {
        "gate": cmd_gate,
        "extract": cmd_extract,
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
