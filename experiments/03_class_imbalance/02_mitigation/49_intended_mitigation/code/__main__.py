"""Command-line entry point for the intended-mitigation study (exp-49)."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401

from centre import N_DRAWS

from intended import load_protocol_config, validate_protocol
from intended.analyze import run_analyze
from intended.fit_lr import fit_lr_shard
from intended.frozen_sgd import fit_sgd_draw
from intended.preflight import run_preflight
from intended.reproduction import check_ce

__all__ = ["main"]

logger = logging.getLogger(__name__)
_LR_STAGE = {"ce": "ce", "lr": "rest"}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        action="append",
        required=True,
        help="study config (repeat for analyze)",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("preflight", help="gates 1, 2, 4 (no fits)")
    fit = commands.add_parser("fit", help="fit one split (all draws) or one draw")
    fit.add_argument("--stage", required=True, choices=("ce", "lr", "sgd"))
    fit.add_argument("--split-index", type=int, choices=(0, 1, 2))
    fit.add_argument("--draw-index", type=int, choices=range(N_DRAWS))
    fit.add_argument("--shard-index", type=int, choices=range(3 * N_DRAWS))
    commands.add_parser("check-ce", help="gate 3: CE refits against exp-25/26")
    commands.add_parser("analyze", help="all contrasts over both configs")
    return parser


def _passed(path: Path) -> bool:
    payload = json.loads(path.read_text(encoding="utf-8"))
    logger.info("%s: passed=%s", path, payload["passed"])
    return bool(payload["passed"])


def main(argv: list[str] | None = None) -> None:
    """Dispatch one subcommand; a failed gate exits non-zero."""
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    configs = [load_protocol_config(Path(p)) for p in args.config]
    for config in configs:
        validate_protocol(config)
    if args.command == "analyze":
        logger.info("analysis written to %s", run_analyze(configs))
        return
    if len(configs) != 1:
        raise SystemExit(f"{args.command} takes exactly one --config")
    config = configs[0]
    if args.command in ("preflight", "check-ce"):
        gate = run_preflight if args.command == "preflight" else check_ce
        if not _passed(gate(config)):
            raise SystemExit(1)
        return
    _fit(config, args)


def _fit(config: dict[str, Any], args: argparse.Namespace) -> None:
    """One SLURM array shard (``split * 10 + draw``), one draw, or every draw of a split."""
    if args.shard_index is not None:
        split, draw = divmod(args.shard_index, N_DRAWS)
        shards = [(split, draw)]
    elif args.split_index is not None:
        draws = range(N_DRAWS) if args.draw_index is None else [args.draw_index]
        shards = [(args.split_index, d) for d in draws]
    else:
        raise SystemExit("fit needs --shard-index or --split-index")
    for split, draw in shards:
        if args.stage == "sgd":
            fit_sgd_draw(config, split, draw)
        else:
            fit_lr_shard(config, split, draw, _LR_STAGE[args.stage])


if __name__ == "__main__":
    main()
