"""Prepend exp-2's and exp-3's code dirs to sys.path (side-effect import, used only
by __main__.py). Needed so ``imbalance_benchmark`` and ``decodability`` (imported as
libraries, never edited) are importable, and so a single
``APPTAINERENV_PYTHONPATH=<this code dir>`` is sufficient on the cluster -- no
per-module path hacking beyond this.
"""

from __future__ import annotations

import sys
from itertools import chain
from pathlib import Path

_EXPERIMENTS = next(p for p in Path(__file__).resolve().parents if p.name == "experiments")
_EXP2_CODE = next(chain(_EXPERIMENTS.glob("*/02_benchmark_patch/code"), _EXPERIMENTS.glob("*/*/02_benchmark_patch/code")))
_EXP3_CODE = next(chain(_EXPERIMENTS.glob("*/03_classifier_limitation/code"), _EXPERIMENTS.glob("*/*/03_classifier_limitation/code")))
for _code_dir in (_EXP2_CODE, _EXP3_CODE):
    if str(_code_dir) not in sys.path:
        sys.path.insert(0, str(_code_dir))
