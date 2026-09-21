"""Prepend earlier experiment code dirs to sys.path."""

from __future__ import annotations

import sys
from pathlib import Path

_EXPERIMENTS = Path(__file__).resolve().parents[3]
_EXP2_CODE = next(_EXPERIMENTS.glob(f"*/02_benchmark_patch/code"))
_EXP3_CODE = next(_EXPERIMENTS.glob(f"*/03_classifier_limitation/code"))
_EXP4_CODE = next(_EXPERIMENTS.glob(f"*/04_patient_influence/code"))
_EXP5_CODE = next(_EXPERIMENTS.glob(f"*/05_effective_support/code"))
_EXP6_CODE = next(_EXPERIMENTS.glob(f"*/06_multidirectional_redundancy/code"))
_EXP7_CODE = next(_EXPERIMENTS.glob(f"*/07_site_coverage/code"))
_EXP8_CODE = next(_EXPERIMENTS.glob(f"*/08_patient_coverage/code"))
_EXP9_CODE = next(_EXPERIMENTS.glob(f"*/09_concentrated_coverage/code"))
_EXP10_CODE = next(_EXPERIMENTS.glob(f"*/10_cohort_composition/code"))
_EXP11_CODE = next(_EXPERIMENTS.glob(f"*/11_coverage_redundancy/code"))

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
):
    if str(_code_dir) not in sys.path:
        sys.path.insert(0, str(_code_dir))
