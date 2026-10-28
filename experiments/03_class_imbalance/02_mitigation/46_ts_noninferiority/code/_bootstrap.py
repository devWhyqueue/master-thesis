"""Make experiment 40's code (``mitigation``, ``analyze``) and its dependencies importable."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

_OWN_CODE = Path(__file__).resolve().parent
_METHODS_CODE = _OWN_CODE.parents[1] / "40_methods" / "code"

runpy.run_path(str(_METHODS_CODE / "_bootstrap.py"))
for _path in (_METHODS_CODE, _OWN_CODE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
