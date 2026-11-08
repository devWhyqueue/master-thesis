"""Command-line entry point for the four-condition LoRA decomposition (exp-48)."""

from __future__ import annotations

import argparse
import logging

import _bootstrap  # noqa: F401

from centre.fit import decode_shard_index

from lora_decomposition.analyze import run_analyze
from lora_decomposition.check import check_runs
from lora_decomposition.fit import run_fit
from lora_decomposition.protocol import ARMS, load_protocol_config
from lora_decomposition.submit import submit_analyze, submit_jobs

__all__ = ["main"]

logger = logging.getLogger(__name__)

_MULTI_CONFIG = ("check", "analyze")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("fit", help="train one arm of one (split, draw) block")
    fit.add_argument("--shard-index", type=int, required=True)
    fit.add_argument("--arm", required=True, choices=ARMS)
    check = commands.add_parser("check", help="technical check of a phase")
    check.add_argument("--phase", required=True, choices=("pilot", "main"))
    commands.add_parser("analyze", help="paired analysis of both datasets")
    submit = commands.add_parser("submit", help="submit pending fits or the analysis")
    submit.add_argument(
        "--phase", required=True, choices=("smoke", "pilot", "main", "analyze")
    )
    submit.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Parse arguments and dispatch one subcommand."""
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    configs = [load_protocol_config(path) for path in args.config]
    multi = args.command in _MULTI_CONFIG or getattr(args, "phase", None) == "analyze"
    if not multi and len(configs) != 1:
        raise SystemExit(f"{args.command} takes exactly one --config")
    if args.command == "fit":
        run_fit(configs[0], *decode_shard_index(args.shard_index), args.arm)
    elif args.command == "check":
        logger.info("check written to %s", check_runs(configs, args.phase))
    elif args.command == "analyze":
        logger.info("analysis written to %s", run_analyze(configs))
    elif args.phase == "analyze":
        logger.info("analyze job: %s", submit_analyze(configs, args.dry_run))
    else:
        logger.info("submitted: %s", submit_jobs(configs[0], args.phase, args.dry_run))


if __name__ == "__main__":
    main()
