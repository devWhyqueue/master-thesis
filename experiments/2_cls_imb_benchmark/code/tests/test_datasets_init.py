from __future__ import annotations

import pytest

from imbalance_benchmark.datasets import build_manifest


def test_build_manifest_rejects_unknown_dataset_name() -> None:
    with pytest.raises(ValueError, match="Unknown dataset"):
        build_manifest({"dataset": {"name": "not-a-real-dataset"}})
