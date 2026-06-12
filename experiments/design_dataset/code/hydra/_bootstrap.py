"""Set up sys.path so hydra job modules can import from shared and code roots."""

import sys
from pathlib import Path


def _setup() -> None:
    experiments_root = Path(__file__).resolve().parents[3]
    code_root = Path(__file__).resolve().parents[1]
    for path in (experiments_root / "shared", code_root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


_setup()
