"""Pytest fixtures and environment configuration for the spectrum-regularization tests."""

from __future__ import annotations

import sys
from pathlib import Path

_CODE = Path(__file__).resolve().parents[1] / "code"
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

import _bootstrap  # noqa: E402,F401
import pytest  # noqa: E402
from imbalance_benchmark.datasets.features.cache import reset_feature_bank  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_feature_bank():
    """Reset the process-global feature bank between tests."""
    reset_feature_bank()
    yield
