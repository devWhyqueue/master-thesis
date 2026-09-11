"""Pytest fixtures and environment configuration for the site-coverage tests."""

from __future__ import annotations

import sys
from pathlib import Path

_CODE = Path(__file__).resolve().parents[1] / "code"
_EXP2_CODE = _CODE.parents[1] / "2_benchmark_patch" / "code"
_EXP3_CODE = _CODE.parents[1] / "3_classifier_limitation" / "code"
_EXP4_CODE = _CODE.parents[1] / "4_patient_influence" / "code"
_EXP5_CODE = _CODE.parents[1] / "5_effective_support" / "code"
_EXP6_CODE = _CODE.parents[1] / "6_multidirectional_redundancy" / "code"

for _code_dir in (_CODE, _EXP2_CODE, _EXP3_CODE, _EXP4_CODE, _EXP5_CODE, _EXP6_CODE):
    if str(_code_dir) not in sys.path:
        sys.path.insert(0, str(_code_dir))

import pytest
from imbalance_benchmark.datasets.features.cache import reset_feature_bank


@pytest.fixture(autouse=True)
def _isolated_feature_bank():
    """Reset the process-global feature bank between tests."""
    reset_feature_bank()
    yield
