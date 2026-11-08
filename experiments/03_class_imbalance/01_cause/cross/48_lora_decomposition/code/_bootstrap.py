"""Make experiment 40's code (``mitigation``, ``analyze``) and its dependencies importable."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

_OWN_CODE = Path(__file__).resolve().parent
_CLASS_IMBALANCE = next(p for p in _OWN_CODE.parents if p.name == "03_class_imbalance")
_METHODS_CODE = _CLASS_IMBALANCE / "02_mitigation" / "40_methods" / "code"

runpy.run_path(str(_METHODS_CODE / "_bootstrap.py"))
for _path in (_METHODS_CODE, _OWN_CODE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
