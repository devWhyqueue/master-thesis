"""Command-line entry point for experiment 36: joint mechanism (separation + centre correction)."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Callable

import _bootstrap  # noqa: F401
from imbalance_benchmark.common import load_config

from breadth.slurm import _job, submit_workflow

from joint.analyze import run_analyze
from joint.fitting.fit import main_shard_count, pilot_shard_count, run_fit_shard
from joint.gating.gate import run_gate
from joint.precheck import run_precheck
from joint.report import run_report

__all__ = ["main"]

_STAGES = (
    "precheck",
    "fit_pilot",
    "analyze_pilot",
    "fit_main",
    "analyze_main",
    "report",
    "all",
)


def _parser() -> argparse.ArgumentParser:
    """Create the joint-mechanism command-line argument parser."""
    parser = argparse.ArgumentParser(description="Joint Mechanism Experiment CLI")
    parser.add_argument("--config", required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("precheck")
    fit = sub.add_parser("fit")
    fit.add_argument("--phase", choices=("pilot", "main"), required=True)
    fit.add_argument("--shard-index", type=int, required=True)
    analyze = sub.add_parser("analyze")
    analyze.add_argument("--phase", choices=("pilot", "main"), required=True)
    sub.add_parser("report")
    submit = sub.add_parser("submit")
    submit.add_argument("--stage", choices=_STAGES, required=True)
    submit.add_argument("--dry-run", action="store_true")
    return parser


def cmd_precheck(args: argparse.Namespace) -> None:
    """Verify exp-34's frozen alpha/centres and exp-25's TCGA-UT G=20 controls are ready."""
    run_precheck(load_config(args.config))


def cmd_fit(args: argparse.Namespace) -> None:
    """Fit every setting/arm/control of one (split, draw) shard."""
    run_fit_shard(load_config(args.config), args.phase, args.shard_index)


def cmd_analyze(args: argparse.Namespace) -> None:
    """Pilot: run the six stopping gates. Main: pool B/P/S/R and the primary estimates."""
    config = load_config(args.config)
    if args.phase == "pilot":
        run_gate(config)
    else:
        run_analyze(config)


def cmd_report(args: argparse.Namespace) -> None:
    """Snapshot precheck, pilot diagnostics, and main analysis for the hand-written report."""
    run_report(load_config(args.config))


def _build_stages(config: dict, chained: bool) -> dict:
    """Build every stage's job, chained with afterok dependencies when ``chained``."""
    precheck = _job(config, "precheck", "precheck")
    fit_pilot = _job(
        config,
        "fit_pilot",
        "fit --phase pilot",
        (precheck.name,) if chained else (),
        array_size=pilot_shard_count(),
    )
    analyze_pilot = _job(
        config,
        "analyze_pilot",
        "analyze --phase pilot",
        (fit_pilot.name,) if chained else (),
    )
    fit_main = _job(
        config,
        "fit_main",
        "fit --phase main",
        (analyze_pilot.name,) if chained else (),
        array_size=main_shard_count(),
    )
    analyze_main = _job(
        config,
        "analyze_main",
        "analyze --phase main",
        (fit_main.name,) if chained else (),
    )
    report = _job(config, "report", "report", (analyze_main.name,) if chained else ())
    return {
        "precheck": precheck,
        "fit_pilot": fit_pilot,
        "analyze_pilot": analyze_pilot,
        "fit_main": fit_main,
        "analyze_main": analyze_main,
        "report": report,
    }


def cmd_submit(args: argparse.Namespace) -> None:
    """Submit one stage, or precheck -> pilot -> gate -> main -> analyze -> report for ``all``."""
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
        "report": cmd_report,
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
