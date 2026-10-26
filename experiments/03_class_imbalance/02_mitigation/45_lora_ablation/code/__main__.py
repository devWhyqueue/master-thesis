"""Command-line entry point for the LoRA-omission follow-up (exp-45)."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import _bootstrap  # noqa: F401

from lora_ablation import load_protocol_config
from lora_ablation.analyze import run_analyze
from lora_ablation.fit import fit_cpu_split
from lora_ablation.frozen_cuda import fit_cuda_draw
from lora_ablation.preflight import audit_inputs, write_inventory

__all__ = ["main"]

logger = logging.getLogger(__name__)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        action="append",
        required=True,
        help="follow-up config (repeat for analyze)",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "preflight", help="audit reusable inputs and save the inventory"
    )
    fit = commands.add_parser(
        "fit", help="fit the frozen-encoder candidates of one split"
    )
    fit.add_argument("--stage", required=True, choices=("cpu", "cuda"))
    fit.add_argument("--split-index", type=int, required=True, choices=(0, 1, 2))
    fit.add_argument(
        "--draw-index",
        type=int,
        choices=range(10),
        help="one draw only (CPU pilot); required for cuda",
    )
    commands.add_parser("analyze", help="paired decisions over all configs")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Dispatch one subcommand."""
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    if args.command == "analyze":
        run_analyze([load_protocol_config(Path(p)) for p in args.config])
        return
    if args.command == "fit" and args.stage == "cuda" and args.draw_index is None:
        raise SystemExit("fit --stage cuda needs --draw-index")
    if len(args.config) != 1:
        raise SystemExit(f"{args.command} takes exactly one --config")
    config = load_protocol_config(Path(args.config[0]))
    if args.command == "preflight":
        inventory = audit_inputs(config)
        path = write_inventory(config, inventory)
        logger.info("inventory %s: %s", path, json.dumps(inventory["workload"]))
        return
    if args.stage == "cuda":
        fit_cuda_draw(config, args.split_index, args.draw_index)
    else:
        fit_cpu_split(config, args.split_index, args.draw_index)


if __name__ == "__main__":
    main()
