import sys
from pathlib import Path

# experiments/3_diversity/code (this package) and experiments/2_benchmark_patch/code
# (imported as a library, never edited) must both be importable.
_EXP3_CODE = Path(__file__).resolve().parents[1]
_EXP2_CODE = _EXP3_CODE.parents[1] / "2_benchmark_patch" / "code"
for code_dir in (_EXP3_CODE, _EXP2_CODE):
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))

import pytest

from imbalance_benchmark.datasets.features.cache import reset_feature_bank


@pytest.fixture(autouse=True)
def _isolated_feature_bank():
    """Reset the process-global feature bank between tests.

    Mirrors experiments/2_benchmark_patch/code/tests/conftest.py: tests use
    varying feature dims for the same manifest-derived paths under different
    tmp_path roots, so the bank must not leak across tests.
    """
    reset_feature_bank()
    yield
