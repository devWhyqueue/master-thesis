"""Prepend exp-2's and exp-3's code dirs to sys.path (side-effect import, used only
by __main__.py). Needed so ``imbalance_benchmark`` and ``decodability`` (imported as
libraries, never edited) are importable, and so a single
``APPTAINERENV_PYTHONPATH=<this code dir>`` is sufficient on the cluster -- no
per-module path hacking beyond this.
"""

from __future__ import annotations

import sys
from pathlib import Path

_EXPERIMENTS = Path(__file__).resolve().parents[2]
_EXP2_CODE = _EXPERIMENTS / "2_benchmark_patch" / "code"
_EXP3_CODE = _EXPERIMENTS / "3_classifier_limitation" / "code"
for _code_dir in (_EXP2_CODE, _EXP3_CODE):
    if str(_code_dir) not in sys.path:
        sys.path.insert(0, str(_code_dir))
