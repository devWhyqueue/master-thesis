"""Set up sys.path so hydra job modules can import from shared and code roots."""

import sys
from pathlib import Path


def _setup() -> None:
    here = Path(__file__).resolve()
    # code_root is the parent of the hydra/ package (one level up from this file)
    code_root = here.parent.parent
    # experiments_root is the directory named 'experiments' somewhere above us
    experiments_root = next(
        (p for p in here.parents if p.name == "experiments"),
        here.parents[3],  # ponytail: fallback to old fixed index if walk fails
    )
    for path in (experiments_root / "shared", code_root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


_setup()
