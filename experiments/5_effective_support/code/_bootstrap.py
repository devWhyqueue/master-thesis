"""Prepend earlier experiment code dirs to sys.path."""

from __future__ import annotations

import sys
from pathlib import Path

_EXPERIMENTS = Path(__file__).resolve().parents[2]
_EXP2_CODE = _EXPERIMENTS / "2_benchmark_patch" / "code"
_EXP3_CODE = _EXPERIMENTS / "3_classifier_limitation" / "code"
_EXP4_CODE = _EXPERIMENTS / "4_patient_influence" / "code"

for _code_dir in (_EXP2_CODE, _EXP3_CODE, _EXP4_CODE):
    if str(_code_dir) not in sys.path:
        sys.path.insert(0, str(_code_dir))
