"""Prepend earlier experiment code dirs to sys.path."""

from __future__ import annotations

import sys
from itertools import chain
from pathlib import Path

_EXPERIMENTS = next(p for p in Path(__file__).resolve().parents if p.name == "experiments")
_EXP2_CODE = next(chain(_EXPERIMENTS.glob("*/02_benchmark_patch/code"), _EXPERIMENTS.glob("*/*/02_benchmark_patch/code")))
_EXP3_CODE = next(chain(_EXPERIMENTS.glob("*/03_classifier_limitation/code"), _EXPERIMENTS.glob("*/*/03_classifier_limitation/code")))
_EXP4_CODE = next(chain(_EXPERIMENTS.glob("*/04_patient_influence/code"), _EXPERIMENTS.glob("*/*/04_patient_influence/code")))
_EXP5_CODE = next(chain(_EXPERIMENTS.glob("*/05_effective_support/code"), _EXPERIMENTS.glob("*/*/05_effective_support/code")))
_EXP6_CODE = next(chain(_EXPERIMENTS.glob("*/06_multidirectional_redundancy/code"), _EXPERIMENTS.glob("*/*/06_multidirectional_redundancy/code")))
_EXP7_CODE = next(chain(_EXPERIMENTS.glob("*/07_site_coverage/code"), _EXPERIMENTS.glob("*/*/07_site_coverage/code")))

for _code_dir in (
    _EXP2_CODE,
    _EXP3_CODE,
    _EXP4_CODE,
    _EXP5_CODE,
    _EXP6_CODE,
    _EXP7_CODE,
):
    if str(_code_dir) not in sys.path:
        sys.path.insert(0, str(_code_dir))
