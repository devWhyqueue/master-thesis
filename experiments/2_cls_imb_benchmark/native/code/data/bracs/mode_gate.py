"""Run a command only when the prepared BRACS benchmark mode matches."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse mode-gate arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-report", required=True)
    parser.add_argument("--required-mode", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> None:
    """Run the wrapped Python command if BRACS mode matches."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    mode = _mode(Path(args.prepare_report))
    if mode != args.required_mode:
        logger.info("Skipping command because BRACS mode is %s.", mode)
        return
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise ValueError("Missing wrapped command.")
    subprocess.run([sys.executable, *command], check=True)


def _mode(path: Path) -> str:
    if not path.exists():
        return "power_law"
    report = json.loads(path.read_text(encoding="utf-8"))
    return str(report.get("recommended_benchmark_mode", "native"))


if __name__ == "__main__":
    main()
