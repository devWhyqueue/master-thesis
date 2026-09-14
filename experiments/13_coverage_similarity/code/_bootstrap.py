"""Prepend earlier experiment code dirs to sys.path."""

from __future__ import annotations

import sys
from pathlib import Path

_EXPERIMENTS = Path(__file__).resolve().parents[2]
_EXP2_CODE = _EXPERIMENTS / "02_benchmark_patch" / "code"
_EXP3_CODE = _EXPERIMENTS / "03_classifier_limitation" / "code"
_EXP4_CODE = _EXPERIMENTS / "04_patient_influence" / "code"
_EXP5_CODE = _EXPERIMENTS / "05_effective_support" / "code"
_EXP6_CODE = _EXPERIMENTS / "06_multidirectional_redundancy" / "code"
_EXP7_CODE = _EXPERIMENTS / "07_site_coverage" / "code"
_EXP8_CODE = _EXPERIMENTS / "08_patient_coverage" / "code"
_EXP9_CODE = _EXPERIMENTS / "09_concentrated_coverage" / "code"
_EXP10_CODE = _EXPERIMENTS / "10_cohort_composition" / "code"
_EXP11_CODE = _EXPERIMENTS / "11_coverage_redundancy" / "code"
_EXP12_CODE = _EXPERIMENTS / "12_shortage_selection" / "code"

for _code_dir in (
    _EXP2_CODE,
    _EXP3_CODE,
    _EXP4_CODE,
    _EXP5_CODE,
    _EXP6_CODE,
    _EXP7_CODE,
    _EXP8_CODE,
    _EXP9_CODE,
    _EXP10_CODE,
    _EXP11_CODE,
    _EXP12_CODE,
):
    if str(_code_dir) not in sys.path:
        sys.path.insert(0, str(_code_dir))
