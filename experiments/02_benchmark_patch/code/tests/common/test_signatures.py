from __future__ import annotations

from pathlib import Path

import numpy as np

from imbalance_benchmark.analysis.inference.crossed_permutation import (
    crossed_block_permutation_tail_nll,
)
from imbalance_benchmark.analysis.query import _confirmation_dir
from imbalance_benchmark.modeling.workflows.confirmation_helpers import (
    _environment_payload,
)

def test_test_prediction_hash_is_prediction_sensitive() -> None:
    from imbalance_benchmark.modeling.workflows.confirmation import (
        _test_prediction_hash,
    )

    base = {
        "test": {
            "labels": [0, 1],
            "preds": [0, 1],
            "probabilities": [[0.9, 0.1], [0.2, 0.8]],
        }
    }
    flipped = {
        "test": {
            "labels": [0, 1],
            "preds": [1, 0],
            "probabilities": [[0.9, 0.1], [0.2, 0.8]],
        }
    }

    assert _test_prediction_hash(base) == _test_prediction_hash(base)
    assert _test_prediction_hash(base) != _test_prediction_hash(flipped)

def test_balanced_predictions_use_one_unassigned_result_directory(
    tmp_path: Path,
) -> None:
    """Assignment-specific analyses reuse one balanced record rather than copies."""
    paths = {"results": tmp_path / "results"}
    balanced = paths["results"] / "assignment=unassigned" / "balanced" / "ce"
    balanced.mkdir(parents=True)

    resolved = _confirmation_dir(paths, "balanced", "ce", "reversed")

    assert resolved == balanced
    assert not (paths["results"] / "assignment=reversed" / "balanced").exists()

def test_crossed_tail_permutation_accepts_a_locked_tail_for_each_split() -> None:
    labels = np.array([0, 1, 2, 0, 1, 2])
    probabilities = np.eye(3)[labels]
    methods = np.stack([probabilities, probabilities])
    ce = np.stack([probabilities[[1, 2, 0, 1, 2, 0]], probabilities])
    blocks = [
        (labels, methods, ce, np.array([f"a{index}" for index in range(6)])),
        (labels, methods, ce, np.array([f"b{index}" for index in range(6)])),
    ]

    p_value = crossed_block_permutation_tail_nll(
        blocks, [[2], [1]], n_permutations=32, seed=3
    )

    assert 0.0 <= p_value <= 1.0

def test_environment_payload_records_the_dependency_lock() -> None:
    environment = _environment_payload()

    assert environment["dependency_lock"]["path"] == "uv.lock"
    assert len(environment["dependency_lock"]["sha256"]) == 64
