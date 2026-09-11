"""Command-line entry point for exp-4 class decodability experiment."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Callable

import _bootstrap  # noqa: F401
from imbalance_benchmark.common import load_config

from decodability import SUPPORTS
from decodability import analyze as analyze_stage
from decodability import audit as audit_stage
from decodability import probe as probe_stage
from decodability import select as select_stage
from decodability import slurm as slurm_stage
from decodability.slurm import PROBE_VAL_ARRAY_SIZE


def _parser() -> argparse.ArgumentParser:
    """Create exp-4 command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Exp-4 Class Decodability Under Patient Shortage CLI"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--split-index", type=int, choices=range(3), default=None)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("preflight")

    probe_val = sub.add_parser("probe-val")
    probe_val.add_argument(
        "--shard-index", type=int, choices=range(PROBE_VAL_ARRAY_SIZE), required=True
    )

    sub.add_parser("select")

    probe_test = sub.add_parser("probe-test")
    probe_test.add_argument("--condition", choices=SUPPORTS, required=True)

    sub.add_parser("analyze")

    submit = sub.add_parser("submit")
    submit.add_argument("--dry-run", action="store_true")

    return parser


def cmd_preflight(args: argparse.Namespace) -> None:
    """Run Gate 0 preflight checks and write signed preflight.json."""
    audit_stage.run_preflight(load_config(args.config))


def cmd_probe_val(args: argparse.Namespace) -> None:
    """Run one shard of validation probe fitting/search."""
    probe_stage.run_probe_val(load_config(args.config), args.shard_index)


def cmd_select(args: argparse.Namespace) -> None:
    """Select probe hyperparameters across splits and write signed probe_selection.json."""
    select_stage.run_select(load_config(args.config))


def cmd_probe_test(args: argparse.Namespace) -> None:
    """Run test probe evaluation for specified split and condition."""
    if args.split_index is None:
        raise ValueError("--split-index is required for probe-test")
    probe_stage.run_probe_test(
        load_config(args.config), args.split_index, args.condition
    )


def cmd_analyze(args: argparse.Namespace) -> None:
    """Compute contrasts, generate tables, and plot decodability curves."""
    analyze_stage.run_analyze(load_config(args.config))


def cmd_submit(args: argparse.Namespace) -> None:
    """Render and submit exp-4's SLURM workflow."""
    config = load_config(args.config)
    config_path = str(Path(args.config).resolve())
    slurm_stage.submit_workflow(config, config_path, args.dry_run)


def _commands() -> dict[str, Callable[[argparse.Namespace], None]]:
    """Return CLI dispatch table."""
    return {
        "preflight": cmd_preflight,
        "probe-val": cmd_probe_val,
        "select": cmd_select,
        "probe-test": cmd_probe_test,
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
