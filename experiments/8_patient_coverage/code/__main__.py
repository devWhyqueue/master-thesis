"""Command-line entry point for the patient-coverage experiment."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Callable

import _bootstrap  # noqa: F401
from imbalance_benchmark.common import load_config

from breadth.slurm import _job, submit_workflow

from neighbours import FIT_SHARD_COUNT
from neighbours import analyze as analyze_stage
from neighbours import census as census_stage
from neighbours import stages as stages_stage


def _parser() -> argparse.ArgumentParser:
    """Create the patient-coverage command-line argument parser."""
    parser = argparse.ArgumentParser(description="Patient Coverage Experiment CLI")
    parser.add_argument("--config", required=True)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("census")

    fit_parser = sub.add_parser("fit")
    fit_parser.add_argument(
        "--shard-index", type=int, choices=range(FIT_SHARD_COUNT), required=True
    )

    sub.add_parser("analyze")

    submit = sub.add_parser("submit")
    submit.add_argument("--dry-run", action="store_true")

    return parser


def cmd_census(args: argparse.Namespace) -> None:
    """Census the patient-coverage pool and gate the fit array on it."""
    census_stage.run_census(load_config(args.config))


def cmd_fit(args: argparse.Namespace) -> None:
    """Run one (split, allocation) shard of the patient-coverage fits."""
    stages_stage.run_fit_shard(load_config(args.config), args.shard_index)


def cmd_analyze(args: argparse.Namespace) -> None:
    """Compute the coverage gain, within-neighbourhood residual, and their interpretation."""
    analyze_stage.run_analyze(load_config(args.config))


def cmd_submit(args: argparse.Namespace) -> None:
    """Render and submit the experiment's SLURM workflow: census -> fit -> analyze."""
    config = load_config(args.config)
    config_path = str(Path(args.config).resolve())
    census = _job(config, "census", "census")
    fit = _job(
        config, "fit", "fit", dependencies=(census.name,), array_size=FIT_SHARD_COUNT
    )
    analyze = _job(config, "analyze", "analyze", dependencies=(fit.name,))
    submit_workflow(config, config_path, args.dry_run, jobs=[census, fit, analyze])


def _commands() -> dict[str, Callable[[argparse.Namespace], None]]:
    """Return CLI dispatch table."""
    return {
        "census": cmd_census,
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
