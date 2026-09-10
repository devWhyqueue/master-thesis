"""Unit tests for model fitting and shard decoding."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from breadth import (
    BREADTH_LADDER,
    DEPTH_LADDER,
    FIT_SHARD_COUNT,
    GRID_CELLS,
    N_SPLITS,
)
from breadth.fit import EvalPartition, decode_shard_index, tune_and_fit_draw


def test_decode_shard_index_covers_entire_grid():
    """All 27 shards map to unique (split, g, m) combinations."""
    seen = set()
    for shard_idx in range(FIT_SHARD_COUNT):
        split_idx, g, m = decode_shard_index(shard_idx)
        assert split_idx in range(N_SPLITS)
        assert (g, m) in GRID_CELLS
        seen.add((split_idx, g, m))

    assert len(seen) == FIT_SHARD_COUNT
    assert seen == {
        (s, g, m)
        for s in range(N_SPLITS)
        for g in BREADTH_LADDER
        for m in DEPTH_LADDER
    }


def test_tune_and_fit_draw_synthetic():
    """Validation tuning selects converging lambda and produces test predictions."""
    rng = np.random.default_rng(42)
    dim = 16
    n_train = 60
    n_val = 30
    n_test = 30

    train_x = rng.normal(size=(n_train, dim))
    train_y = rng.choice([0, 1, 2], size=n_train)

    val_x = rng.normal(size=(n_val, dim))
    val_y = rng.choice([0, 1, 2], size=n_val)
    val_id = pd.DataFrame(
        {
            "case_id": [f"val_case_{i // 5}" for i in range(n_val)],
            "slide_id": [f"val_slide_{i}" for i in range(n_val)],
        }
    )

    test_x = rng.normal(size=(n_test, dim))
    test_y = rng.choice([0, 1, 2], size=n_test)
    test_id = pd.DataFrame(
        {
            "case_id": [f"test_case_{i // 5}" for i in range(n_test)],
            "slide_id": [f"test_slide_{i}" for i in range(n_test)],
        }
    )

    evals = EvalPartition(val_x, val_y, val_id, test_x, test_y, test_id)
    (
        fit_res,
        best_lam,
        test_preds,
        test_probs,
        val_end,
        test_end,
    ) = tune_and_fit_draw(train_x, train_y, evals)

    assert fit_res.converged is True
    assert best_lam > 0
    assert len(test_preds) == n_test
    assert test_probs.shape == (n_test, 3)
    assert "patient_macro_balanced_accuracy" in test_end
