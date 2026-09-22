"""Pytest fixtures and environment configuration for exp-4 tests."""

from __future__ import annotations

import sys
from itertools import chain
from pathlib import Path

_CODE = Path(__file__).resolve().parents[1] / "code"
_EXPERIMENTS = next(p for p in _CODE.parents if p.name == "experiments")
_EXP2_CODE = next(chain(_EXPERIMENTS.glob("*/02_benchmark_patch/code"), _EXPERIMENTS.glob("*/*/02_benchmark_patch/code")))
for code_dir in (_CODE, _EXP2_CODE):
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))

import pytest
from imbalance_benchmark.datasets.features.cache import reset_feature_bank


@pytest.fixture(autouse=True)
def _isolated_feature_bank():
    """Reset the process-global feature bank between tests."""
    reset_feature_bank()
    yield
