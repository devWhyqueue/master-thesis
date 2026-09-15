"""Command-line entry point for the hull-coverage experiment."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Callable

import _bootstrap  # noqa: F401
from imbalance_benchmark.common import load_config

from breadth.slurm import _job, submit_workflow

from decomposition.precision import load_precision

from hull import N_SPLITS
from hull.census import run_census_shard, run_recheck
from hull.fit import run_fit_shard, shard_count
from hull.inference.analyze import run_analyze
from hull.inference.precision import run_precision

__all__ = ["main"]

_STAGES = ("census", "precision", "fit", "analyze")


def _parser() -> argparse.ArgumentParser:
    """Create the hull-coverage command-line argument parser."""
    parser = argparse.ArgumentParser(description="Hull Coverage Experiment CLI")
    parser.add_argument("--config", required=True)
    sub = parser.add_subparsers(dest="command", required=True)

    census = sub.add_parser("census")
    census.add_argument(
        "--shard-index", type=int, choices=range(N_SPLITS), required=True
    )
    sub.add_parser("recheck")
    sub.add_parser("precision")
    fit = sub.add_parser("fit")
    fit.add_argument("--shard-index", type=int, required=True)
    sub.add_parser("analyze")

    submit = sub.add_parser("submit")
    submit.add_argument("--stage", choices=_STAGES, required=True)
    submit.add_argument("--dry-run", action="store_true")
    return parser


def _selected_draws(config: dict) -> int:
    n_draws = load_precision(config)["selected_draws"]
    if not n_draws:
        raise RuntimeError("precision.json has no selected_draws")
    return int(n_draws)


def cmd_census(args: argparse.Namespace) -> None:
    """Search one split's cohorts and run its manipulation check."""
    run_census_shard(load_config(args.config), args.shard_index)


def cmd_recheck(args: argparse.Namespace) -> None:
    """Re-evaluate the manipulation check on the stored census cohorts."""
    run_recheck(load_config(args.config))


def cmd_precision(args: argparse.Namespace) -> None:
    """Simulate the class-recall model and select the draw count."""
    run_precision(load_config(args.config))


def cmd_fit(args: argparse.Namespace) -> None:
    """Fit one (split, five-draw block) shard of uniform-patient-count classifiers."""
    config = load_config(args.config)
    run_fit_shard(config, args.shard_index, _selected_draws(config))


def cmd_analyze(args: argparse.Namespace) -> None:
    """Compute the class-recall model, parts, absorption, and the 10-to-20 prediction."""
    run_analyze(load_config(args.config))


def cmd_submit(args: argparse.Namespace) -> None:
    """Submit one gated stage; each stage is submitted only after the previous one passed."""
    config = load_config(args.config)
    sizes = {"census": N_SPLITS, "precision": 0, "analyze": 0}
    array_size = (
        sizes[args.stage]
        if args.stage in sizes
        else shard_count(_selected_draws(config))
    )
    job = _job(config, args.stage, args.stage, array_size=array_size)
    submit_workflow(config, str(Path(args.config).resolve()), args.dry_run, jobs=[job])


def _commands() -> dict[str, Callable[[argparse.Namespace], None]]:
    """Return CLI dispatch table."""
    return {
        "census": cmd_census,
        "recheck": cmd_recheck,
        "precision": cmd_precision,
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
