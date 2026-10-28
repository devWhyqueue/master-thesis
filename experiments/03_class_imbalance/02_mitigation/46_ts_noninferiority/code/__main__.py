"""Command-line entry point for the temperature-scaling noninferiority test (exp-46)."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import _bootstrap  # noqa: F401

from ts_nll import load_protocol
from ts_nll.analyze import run_analyze, run_gate, run_render

logger = logging.getLogger(__name__)

__all__ = ["main"]


def main(argv: list[str] | None = None) -> None:
    """Dispatch ``gate`` (reproduction check), ``analyze`` (contrasts and verdict) or ``render`` (results.tex)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("command", choices=("gate", "analyze", "render"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    if args.command == "render":
        logger.info(run_render())
        return
    protocol = load_protocol(Path(args.config))
    if args.command == "gate":
        run_gate(protocol)
    else:
        logger.info(run_analyze(protocol))


if __name__ == "__main__":
    main()
