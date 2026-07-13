"""Command-line entry point for the unified imbalance benchmark."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable

from imbalance_benchmark.commands import (
    cmd_analyze,
    cmd_confirm,
    cmd_freeze,
    cmd_pilot,
    cmd_prepare,
    cmd_smoke,
    cmd_submit,
    cmd_tune,
)


def _parser() -> argparse.ArgumentParser:
    """Create the benchmark command-line parser."""
    parser = argparse.ArgumentParser(description="Class-Imbalance Benchmark CLI")
    parser.add_argument("--config", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-index", type=int, choices=range(3), default=None)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in (
        "prepare",
        "pilot",
        "freeze",
        "tune",
        "confirm",
        "analyze",
        "smoke",
    ):
        sub.add_parser(command)
    submit = sub.add_parser("submit")
    submit.add_argument("--dry-run", action="store_true")
    submit.add_argument("--smoke", action="store_true")
    for command in ("tune", "confirm"):
        sub.choices[command].add_argument(
            "--condition", choices=("balanced", "moderate", "severe")
        )
    return parser


def _commands() -> dict[str, Callable[[argparse.Namespace], None]]:
    """Return CLI command handlers keyed by their subcommand names."""
    return {
        "prepare": cmd_prepare,
        "pilot": cmd_pilot,
        "freeze": cmd_freeze,
        "tune": cmd_tune,
        "confirm": cmd_confirm,
        "analyze": cmd_analyze,
        "submit": cmd_submit,
        "smoke": cmd_smoke,
    }


def main() -> None:
    """Parse benchmark command-line arguments and dispatch the selected command."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    args = _parser().parse_args()
    _commands()[args.command](args)


if __name__ == "__main__":
    main()
