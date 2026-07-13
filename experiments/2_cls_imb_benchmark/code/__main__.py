from __future__ import annotations

import argparse
import logging

from imbalance_benchmark.commands import (
    cmd_prepare,
    cmd_pilot,
    cmd_freeze,
    cmd_tune,
    cmd_confirm,
    cmd_analyze,
    cmd_submit,
    cmd_smoke,
)


def main() -> None:
    """CLI execution entrypoint."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(description="Class-Imbalance Benchmark CLI")
    parser.add_argument("--config", default=None)
    parser.add_argument("--seed", type=int, default=0)
    sub = parser.add_subparsers(dest="command", required=True)
    for c in ["prepare", "pilot", "freeze", "tune", "confirm", "analyze", "smoke"]:
        sub.add_parser(c)
    sub.add_parser("submit").add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cmds = {
        "prepare": cmd_prepare,
        "pilot": cmd_pilot,
        "freeze": cmd_freeze,
        "tune": cmd_tune,
        "confirm": cmd_confirm,
        "analyze": cmd_analyze,
        "submit": cmd_submit,
        "smoke": cmd_smoke,
    }
    cmds[args.command](args)


if __name__ == "__main__":
    main()
