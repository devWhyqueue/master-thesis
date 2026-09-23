"""Command-line entry point for the rate-weighted worst tail-class assignment experiment."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Callable

import _bootstrap  # noqa: F401
from imbalance_benchmark.common import load_config

from breadth.slurm import _job, submit_workflow

from rate.analyze import run_analyze
from rate.fit import run_fit_shard, shard_count
from rate.intuition import score_intuition_figure
from rate.precheck import run_precheck

__all__ = ["main"]

_STAGES = ("precheck", "fit", "analyze", "all")


def _parser() -> argparse.ArgumentParser:
    """Create the rate-weighted worst-assignment command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Rate-Weighted Worst Tail-Class Assignment Experiment CLI"
    )
    parser.add_argument("--config", required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("precheck")
    fit = sub.add_parser("fit")
    fit.add_argument("--shard-index", type=int, required=True)
    sub.add_parser("analyze")
    sub.add_parser("intuition")
    submit = sub.add_parser("submit")
    submit.add_argument("--stage", choices=_STAGES, required=True)
    submit.add_argument("--dry-run", action="store_true")
    return parser


def cmd_precheck(args: argparse.Namespace) -> None:
    """Run Part A's gate (0 new fits), fit the calibration line, and write precheck.json."""
    run_precheck(load_config(args.config))


def cmd_fit(args: argparse.Namespace) -> None:
    """Fit every pending worst/flip/mild arm of one (split, draw) shard."""
    run_fit_shard(load_config(args.config), args.shard_index)


def cmd_analyze(args: argparse.Namespace) -> None:
    """Pool reused and new arm accuracy, compute H1-H3 and calibration, and write the report."""
    run_analyze(load_config(args.config))


def cmd_intuition(args: argparse.Namespace) -> None:
    """Render the Design-section score-intuition figure from the committed diagnostics.json."""
    del args
    report_dir = Path(__file__).resolve().parents[1] / "report"
    score_intuition_figure(
        report_dir / "diagnostics.json", report_dir / "score_intuition_r100.pdf"
    )


def cmd_submit(args: argparse.Namespace) -> None:
    """Submit one stage, or fit -> analyze chained with afterok for ``all``."""
    config = load_config(args.config)
    chained = args.stage == "all"
    precheck = _job(config, "precheck", "precheck")
    fit = _job(config, "fit", "fit", array_size=shard_count())
    analyze = _job(config, "analyze", "analyze", ("fit",) if chained else ())
    stages = {"precheck": precheck, "fit": fit, "analyze": analyze}
    jobs = [fit, analyze] if chained else [stages[args.stage]]
    submit_workflow(config, str(Path(args.config).resolve()), args.dry_run, jobs=jobs)


def _commands() -> dict[str, Callable[[argparse.Namespace], None]]:
    """Return CLI dispatch table."""
    return {
        "precheck": cmd_precheck,
        "fit": cmd_fit,
        "analyze": cmd_analyze,
        "intuition": cmd_intuition,
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
