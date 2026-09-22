"""Prepend exp-2's code dir to sys.path (side-effect import, used only by __main__.py).

Needed so ``imbalance_benchmark`` (imported as a library, never edited) is
importable, and so a single ``APPTAINERENV_PYTHONPATH=<this code dir>`` is
sufficient on the cluster -- no per-module path hacking beyond this.
"""

from __future__ import annotations

import sys
from itertools import chain
from pathlib import Path

_EXPERIMENTS = next(
    p for p in Path(__file__).resolve().parents if p.name == "experiments"
)
_EXP2_CODE = next(
    chain(
        _EXPERIMENTS.glob("*/02_benchmark_patch/code"),
        _EXPERIMENTS.glob("*/*/02_benchmark_patch/code"),
    )
)
if str(_EXP2_CODE) not in sys.path:
    sys.path.insert(0, str(_EXP2_CODE))
