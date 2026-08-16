import sys
from pathlib import Path

# Add experiments/2_benchmark_patch/code to sys.path so imbalance_benchmark is importable
CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

import pytest

from imbalance_benchmark.datasets.features.cache import reset_feature_bank


@pytest.fixture(autouse=True)
def _isolated_feature_bank():
    """Reset the process-global feature bank between tests.

    Tests use varying feature dims for the same manifest-derived paths under
    different tmp_path roots; a real run only ever seeds one bank.
    """
    reset_feature_bank()
    yield
