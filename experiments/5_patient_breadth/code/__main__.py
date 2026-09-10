"""Command-line entry point for the patient-breadth experiment."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Callable

import _bootstrap  # noqa: F401
from imbalance_benchmark.common import load_config

from breadth import FIT_SHARD_COUNT
from breadth import analyze as analyze_stage
from breadth import audit as audit_stage
from breadth import fit as fit_stage
from breadth import slurm as slurm_stage


def _parser() -> argparse.ArgumentParser:
    """Create the patient-breadth command-line argument parser."""
    parser = argparse.ArgumentParser(description="Patient Breadth Experiment CLI")
    parser.add_argument("--config", required=True)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("preflight")

    fit_parser = sub.add_parser("fit")
    fit_parser.add_argument(
        "--shard-index", type=int, choices=range(FIT_SHARD_COUNT), required=True
    )

    sub.add_parser("analyze")

    submit = sub.add_parser("submit")
    submit.add_argument("--dry-run", action="store_true")

    return parser


def cmd_preflight(args: argparse.Namespace) -> None:
    """Run preflight checks and write signed preflight.json."""
    audit_stage.run_preflight(load_config(args.config))


def cmd_fit(args: argparse.Namespace) -> None:
    """Run one shard of the patient-breadth logistic fits across draws."""
    fit_stage.run_fit_shard(load_config(args.config), args.shard_index)


def cmd_analyze(args: argparse.Namespace) -> None:
    """Compute contrasts, generate tables, and plot support surfaces."""
    analyze_stage.run_analyze(load_config(args.config))


def cmd_submit(args: argparse.Namespace) -> None:
    """Render and submit the experiment's SLURM workflow."""
    config = load_config(args.config)
    config_path = str(Path(args.config).resolve())
    slurm_stage.submit_workflow(config, config_path, args.dry_run)


def _commands() -> dict[str, Callable[[argparse.Namespace], None]]:
    """Return CLI dispatch table."""
    return {
        "preflight": cmd_preflight,
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

