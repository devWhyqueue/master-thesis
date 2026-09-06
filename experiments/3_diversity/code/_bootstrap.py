"""Prepend exp-2's code dir to sys.path (side-effect import, used only by __main__.py).

Needed so ``imbalance_benchmark`` (imported as a library, never edited) and
the top-level ``derive_deficit_thresholds`` script are importable, and so a
single ``APPTAINERENV_PYTHONPATH=<this code dir>`` is sufficient on the
cluster (plan "Cluster wiring") -- no per-module path hacking beyond this.
"""

from __future__ import annotations

import sys
from pathlib import Path

_EXP2_CODE = Path(__file__).resolve().parents[2] / "2_benchmark_patch" / "code"
if str(_EXP2_CODE) not in sys.path:
    sys.path.insert(0, str(_EXP2_CODE))
