"""Pytest fixtures and environment configuration for the concentrated-coverage tests."""

from __future__ import annotations

import sys
from itertools import chain
from pathlib import Path

_CODE = Path(__file__).resolve().parents[1] / "code"
_EXPERIMENTS = next(p for p in _CODE.parents if p.name == "experiments")
_EXP2_CODE = next(chain(_EXPERIMENTS.glob("*/02_benchmark_patch/code"), _EXPERIMENTS.glob("*/*/02_benchmark_patch/code")))
_EXP3_CODE = next(chain(_EXPERIMENTS.glob("*/03_classifier_limitation/code"), _EXPERIMENTS.glob("*/*/03_classifier_limitation/code")))
_EXP4_CODE = next(chain(_EXPERIMENTS.glob("*/04_patient_influence/code"), _EXPERIMENTS.glob("*/*/04_patient_influence/code")))
_EXP5_CODE = next(chain(_EXPERIMENTS.glob("*/05_effective_support/code"), _EXPERIMENTS.glob("*/*/05_effective_support/code")))
_EXP6_CODE = next(chain(_EXPERIMENTS.glob("*/06_multidirectional_redundancy/code"), _EXPERIMENTS.glob("*/*/06_multidirectional_redundancy/code")))
_EXP7_CODE = next(chain(_EXPERIMENTS.glob("*/07_site_coverage/code"), _EXPERIMENTS.glob("*/*/07_site_coverage/code")))
_EXP8_CODE = next(chain(_EXPERIMENTS.glob("*/08_patient_coverage/code"), _EXPERIMENTS.glob("*/*/08_patient_coverage/code")))

for _code_dir in (
    _CODE,
    _EXP2_CODE,
    _EXP3_CODE,
    _EXP4_CODE,
    _EXP5_CODE,
    _EXP6_CODE,
    _EXP7_CODE,
    _EXP8_CODE,
):
    if str(_code_dir) not in sys.path:
        sys.path.insert(0, str(_code_dir))

import pytest
from imbalance_benchmark.datasets.features.cache import reset_feature_bank


@pytest.fixture(autouse=True)
def _isolated_feature_bank():
    """Reset the process-global feature bank between tests."""
    reset_feature_bank()
    yield
