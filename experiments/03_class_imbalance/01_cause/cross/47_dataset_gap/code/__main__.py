"""Command-line entry point for the dataset-gap descriptive analysis (exp-47)."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import _bootstrap  # noqa: F401
from imbalance_benchmark.common import load_config

from breadth.slurm import _job, submit_workflow

from centre import N_SPLITS

from gap.analysis import run_analyze, run_smoke
from gap.embedding import run_geometry
from gap.plots import build_report_assets
from gap.images import run_images

__all__ = ["main"]

_REPORT = Path(__file__).resolve().parents[1] / "report"
_STAGES = ("smoke", "geometry", "analyze", "images", "all")


def _parser() -> argparse.ArgumentParser:
    """Create the exp-47 command-line argument parser."""
    parser = argparse.ArgumentParser(description="Dataset-Gap Descriptive Analysis CLI")
    parser.add_argument("--config", help="dataset config (all commands except figures)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("smoke")
    geometry = sub.add_parser("geometry")
    geometry.add_argument("--shard-index", type=int, required=True)
    sub.add_parser("analyze")
    sub.add_parser("images")
    sub.add_parser("figures", help="build report figures from report/analysis_*.json")
    submit = sub.add_parser("submit")
    submit.add_argument("--stage", choices=_STAGES, required=True)
    submit.add_argument("--dry-run", action="store_true")
    return parser


def _submit(config: dict, config_path: str, stage: str, dry_run: bool) -> None:
    """Submit one stage, or geometry (one task per split) -> analyze for ``all``."""
    geometry = _job(config, "geometry", "geometry", array_size=N_SPLITS)
    analyze = _job(
        config, "analyze", "analyze", (geometry.name,) if stage == "all" else ()
    )
    jobs = {
        "smoke": [_job(config, "smoke", "smoke")],
        "geometry": [geometry],
        "analyze": [analyze],
        "images": [_job(config, "images", "images")],
        "all": [geometry, analyze],
    }[stage]
    submit_workflow(config, str(Path(config_path).resolve()), dry_run, jobs=jobs)


def main() -> None:
    """Parse command-line arguments and dispatch subcommand."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = _parser()
    args = parser.parse_args()
    if args.command == "figures":
        build_report_assets(_REPORT)
        return
    if not args.config:
        parser.error("--config is required")
    config = load_config(args.config)
    if args.command == "smoke":
        run_smoke(config)
    elif args.command == "geometry":
        run_geometry(config, args.shard_index)
    elif args.command == "analyze":
        run_analyze(config)
    elif args.command == "images":
        run_images(config)
    else:
        _submit(config, args.config, args.stage, args.dry_run)


if __name__ == "__main__":
    main()
