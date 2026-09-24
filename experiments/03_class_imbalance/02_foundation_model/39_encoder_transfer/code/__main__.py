"""Command-line entry point for exp-39's encoder-transfer schedule (phase 01) and
UNI2-h feature extraction/audit (phase 03)."""

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

from transfer import MAIN_DRAWS, extract, manifest
from transfer.schedule import draw_schedule, load_train_identity

logger = logging.getLogger(__name__)

__all__ = ["main"]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Exp-39 encoder-transfer schedule CLI")
    parser.add_argument("--config", required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("schedule")
    p_extract = sub.add_parser("extract-features")
    p_extract.add_argument("--shard-index", type=int, required=True)
    p_extract.add_argument("--shards", type=int, required=True)
    p_extract.add_argument("--dtype", default="float32", choices=["float32", "float16"])
    sub.add_parser("merge-features")
    sub.add_parser("audit-features")
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


def cmd_extract_features(args: argparse.Namespace) -> None:
    """Extract this shard's assigned, not-yet-cached UNI2-h slide tensors."""
    config = load_config(args.config)
    extract.extract_shard(config, args.shard_index, args.shards, dtype=args.dtype)
    logger.info(f"shard {args.shard_index}/{args.shards} done")


def cmd_merge_features(args: argparse.Namespace) -> None:
    """One-time merge of every completed shard's pending UNI2-h records."""
    extract.merge_features(load_config(args.config))
    logger.info("merge complete")


def cmd_audit_features(args: argparse.Namespace) -> None:
    """Verify both encoders' caches and publish feature_audit.json + manifests."""
    audit = manifest.run_audit(load_config(args.config))
    logger.info(
        f"uni2h missing={audit['uni2h']['unresolved_missing']} "
        f"corrupt={audit['uni2h']['unresolved_corrupt']}; "
        f"virchow2 missing={audit['virchow2']['unresolved_missing']} "
        f"corrupt={audit['virchow2']['unresolved_corrupt']}"
    )


def _commands() -> dict[str, Callable[[argparse.Namespace], None]]:
    return {
        "schedule": cmd_schedule,
        "extract-features": cmd_extract_features,
        "merge-features": cmd_merge_features,
        "audit-features": cmd_audit_features,
    }


def main() -> None:
    """Parse command-line arguments and dispatch subcommand."""
    args = _parser().parse_args()
    _commands()[args.command](args)


if __name__ == "__main__":
    main()
