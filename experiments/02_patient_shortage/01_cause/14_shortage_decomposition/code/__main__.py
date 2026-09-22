"""Command-line entry point for the shortage-decomposition experiment."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Callable

import _bootstrap  # noqa: F401
from imbalance_benchmark.common import load_config, output_root

from breadth.slurm import _job, submit_workflow

from decomposition import N_DRAWS_DEFAULT, N_SPLITS
from decomposition.analyze import run_analyze
from decomposition.census import run_census_shard
from decomposition.fit import run_fit_shard, shard_count
from decomposition.precision import load_precision, run_precision

__all__ = ["main"]


def _parser() -> argparse.ArgumentParser:
    """Create the shortage-decomposition command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Shortage Decomposition Experiment CLI"
    )
    parser.add_argument("--config", required=True)
    sub = parser.add_subparsers(dest="command", required=True)

    census = sub.add_parser("census")
    census.add_argument(
        "--shard-index", type=int, choices=range(N_SPLITS), required=True
    )

    sub.add_parser("precision")

    fit = sub.add_parser("fit")
    fit.add_argument("--shard-index", type=int, required=True)

    sub.add_parser("analyze")

    submit = sub.add_parser("submit")
    submit.add_argument("--dry-run", action="store_true")

    return parser


def cmd_census(args: argparse.Namespace) -> None:
    """Search one split's cohorts and run its manipulation check."""
    run_census_shard(load_config(args.config), args.shard_index)


def cmd_precision(args: argparse.Namespace) -> None:
    """Simulate the class-recall model and select the fresh-draw count."""
    run_precision(load_config(args.config))


def cmd_fit(args: argparse.Namespace) -> None:
    """Run one (split, 5-draw block) shard of the shortage-decomposition fits."""
    config = load_config(args.config)
    n_draws = load_precision(config)["selected_draws"]
    if not n_draws:
        raise RuntimeError("precision.json has no selected_draws; cannot fit")
    run_fit_shard(config, args.shard_index, n_draws)


def cmd_analyze(args: argparse.Namespace) -> None:
    """Compute the class-recall model, the three parts, and their calibration."""
    run_analyze(load_config(args.config))


def _submit_gates(config: dict, config_path: str, dry_run: bool) -> None:
    """First submission: census (array over splits) -> precision."""
    census = _job(config, "census", "census", array_size=N_SPLITS)
    precision = _job(config, "precision", "precision", dependencies=(census.name,))
    submit_workflow(config, config_path, dry_run, jobs=[census, precision])


def _submit_fit(config: dict, config_path: str, dry_run: bool, n_draws: int) -> None:
    """After a draw count is selected: census again only if D > 20, then fit -> analyze."""
    jobs = []
    dependencies: tuple[str, ...] = ()
    if n_draws > N_DRAWS_DEFAULT:
        census = _job(config, "census", "census", array_size=N_SPLITS)
        jobs.append(census)
        dependencies = (census.name,)
    fit = _job(
        config, "fit", "fit", dependencies=dependencies, array_size=shard_count(n_draws)
    )
    analyze = _job(config, "analyze", "analyze", dependencies=(fit.name,))
    jobs += [fit, analyze]
    submit_workflow(config, config_path, dry_run, jobs=jobs)


def cmd_submit(args: argparse.Namespace) -> None:
    """Render and submit the next stage(s) of the workflow."""
    config = load_config(args.config)
    config_path = str(Path(args.config).resolve())
    precision_p = output_root(config) / "data" / "precision.json"

    if not precision_p.exists():
        _submit_gates(config, config_path, args.dry_run)
        return

    n_draws = load_precision(config)["selected_draws"]
    if not n_draws:
        raise RuntimeError("precision.json has no selected_draws; nothing to submit")
    _submit_fit(config, config_path, args.dry_run, n_draws)


def _commands() -> dict[str, Callable[[argparse.Namespace], None]]:
    """Return CLI dispatch table."""
    return {
        "census": cmd_census,
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
