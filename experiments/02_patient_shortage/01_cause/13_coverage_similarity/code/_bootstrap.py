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
_EXP8_CODE = next(chain(_EXPERIMENTS.glob("*/08_patient_coverage/code"), _EXPERIMENTS.glob("*/*/08_patient_coverage/code")))
_EXP9_CODE = next(chain(_EXPERIMENTS.glob("*/09_concentrated_coverage/code"), _EXPERIMENTS.glob("*/*/09_concentrated_coverage/code")))
_EXP10_CODE = next(chain(_EXPERIMENTS.glob("*/10_cohort_composition/code"), _EXPERIMENTS.glob("*/*/10_cohort_composition/code")))
_EXP11_CODE = next(chain(_EXPERIMENTS.glob("*/11_coverage_redundancy/code"), _EXPERIMENTS.glob("*/*/11_coverage_redundancy/code")))
_EXP12_CODE = next(chain(_EXPERIMENTS.glob("*/12_shortage_selection/code"), _EXPERIMENTS.glob("*/*/12_shortage_selection/code")))

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
