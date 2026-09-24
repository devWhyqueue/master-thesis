"""Command-line entry point for exp-39's encoder-transfer schedule (phase 01)."""

from __future__ import annotations

import argparse
import logging
from typing import Callable

import _bootstrap  # noqa: F401
from imbalance_benchmark.common import (
    N_PATIENT_SPLITS,
    load_config,
    output_root,
    sign_file,
    write_json,
)

from prevalence import patients_per_class

from transfer import MAIN_DRAWS
from transfer.schedule import draw_schedule, load_train_identity

logger = logging.getLogger(__name__)

__all__ = ["main"]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Exp-39 encoder-transfer schedule CLI")
    parser.add_argument("--config", required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("schedule")
    return parser


def cmd_schedule(args: argparse.Namespace) -> None:
    """Build and hash this dataset's frozen draws-10-19 schedule."""
    config = load_config(args.config)
    g = patients_per_class(config)
    cells = []
    for split_idx in range(N_PATIENT_SPLITS):
        train_df, names = load_train_identity(config, split_idx)
        for draw_idx in MAIN_DRAWS:
            cells.append(draw_schedule(train_df, names, split_idx, draw_idx, g))
    out_path = output_root(config) / "schedule.json"
    write_json(out_path, {"dataset": config["dataset"]["name"], "cells": cells})
    sign_file(out_path)
    logger.info(f"wrote {out_path}")


def _commands() -> dict[str, Callable[[argparse.Namespace], None]]:
    return {"schedule": cmd_schedule}


def main() -> None:
    """Parse command-line arguments and dispatch subcommand."""
    args = _parser().parse_args()
    _commands()[args.command](args)


if __name__ == "__main__":
    main()
