"""Command-line entry point for experiment 38: lambda-selection closure of the prior-channel gap."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Callable

import _bootstrap  # noqa: F401
from imbalance_benchmark.common import load_config

from breadth.slurm import _job, submit_workflow

from selection_rule.analyze import run_analyze
from selection_rule.fit import run_fit_shard, shard_count
from selection_rule.precheck import run_precheck
from selection_rule.rules import run_rules_shard

__all__ = ["main"]

_STAGES = ("precheck", "fit", "oracle", "analyze", "all")


def _parser() -> argparse.ArgumentParser:
    """Create the lambda-selection command-line argument parser."""
    parser = argparse.ArgumentParser(description="Lambda Selection Experiment CLI")
    parser.add_argument("--config", required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("precheck")
    fit = sub.add_parser("fit")
    fit.add_argument("--shard-index", type=int, required=True)
    oracle = sub.add_parser("oracle")
    oracle.add_argument("--shard-index", type=int, required=True)
    sub.add_parser("analyze")
    submit = sub.add_parser("submit")
    submit.add_argument("--stage", choices=_STAGES, required=True)
    submit.add_argument("--dry-run", action="store_true")
    return parser


def cmd_precheck(args: argparse.Namespace) -> None:
    """Phase-1 diagnostic: re-predict exp-36's pilot grids, gate on integrity and closure."""
    run_precheck(load_config(args.config))


def cmd_fit(args: argparse.Namespace) -> None:
    """Phase 2: fit native B/P/S/R of one (split, draw) shard, draws 2-9."""
    run_fit_shard(load_config(args.config), args.shard_index)


def cmd_oracle(args: argparse.Namespace) -> None:
    """Phase 2: tuned/oracle/naive/fixed allocations for one (split, draw) shard's own fits."""
    run_rules_shard(load_config(args.config), args.shard_index)


def cmd_analyze(args: argparse.Namespace) -> None:
    """Phase 2: pool per-rule B/P/S/R accuracy; BRACS computes gap/Delta/closure and the label."""
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
    oracle = _job(
        config,
        "oracle",
        "oracle",
        (fit.name,) if chained else (),
        array_size=shard_count(),
    )
    analyze = _job(config, "analyze", "analyze", (oracle.name,) if chained else ())
    return {"precheck": precheck, "fit": fit, "oracle": oracle, "analyze": analyze}


def cmd_submit(args: argparse.Namespace) -> None:
    """Submit one stage, or precheck -> fit -> oracle -> analyze for ``all``."""
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
        "oracle": cmd_oracle,
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
