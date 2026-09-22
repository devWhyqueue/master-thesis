"""Command-line entry point for the multidirectional-redundancy experiment."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Callable

import _bootstrap  # noqa: F401
from imbalance_benchmark.common import load_config

from redundancy import analyze as analyze_stage
from redundancy import correlate as correlate_stage
from redundancy import slurm as slurm_stage


def _parser() -> argparse.ArgumentParser:
    """Create the multidirectional-redundancy command-line argument parser."""
    parser = argparse.ArgumentParser(description="Multidirectional Redundancy CLI")
    parser.add_argument("--config", required=True)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("correlate")
    sub.add_parser("analyze")

    submit = sub.add_parser("submit")
    submit.add_argument("--dry-run", action="store_true")

    return parser


def cmd_correlate(args: argparse.Namespace) -> None:
    """Compute both redundancy measures on the reproduced exp-5 sample."""
    correlate_stage.run_correlate(load_config(args.config))


def cmd_analyze(args: argparse.Namespace) -> None:
    """Refit the support surface under both redundancy measures."""
    analyze_stage.run_analyze(load_config(args.config))


def cmd_submit(args: argparse.Namespace) -> None:
    """Render and submit the experiment's SLURM workflow."""
    config = load_config(args.config)
    config_path = str(Path(args.config).resolve())
    slurm_stage.submit_workflow(config, config_path, args.dry_run)


def _commands() -> dict[str, Callable[[argparse.Namespace], None]]:
    """Return CLI dispatch table."""
    return {
        "correlate": cmd_correlate,
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
