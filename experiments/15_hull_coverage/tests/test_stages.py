"""Unit tests for fit sharding and the analysis's weighted random-cell means."""

from __future__ import annotations

import numpy as np
import pytest

from hull.fit import decode_shard_index, shard_count
from hull.inference.analyze import weighted_mean


def test_shards_cover_every_split_and_draw_block_once():
    """Shard indices map one-to-one onto (split, five-draw block) pairs."""
    n_draws = 35
    decoded = {decode_shard_index(i, n_draws) for i in range(shard_count(n_draws))}
    assert decoded == {(s, b) for s in range(3) for b in range(7)}
    with pytest.raises(ValueError):
        decode_shard_index(shard_count(n_draws), n_draws)


def test_weighted_mean_counts_blocks_by_weight_and_rows_by_mask():
    """A block with weight 2 counts twice; masked-out rows are ignored."""
    values = np.array([[[1.0], [100.0]], [[3.0], [5.0]]])  # (B=2, C=2, R=1)
    mask = np.array([[True, False], [True, True]])
    w = np.array([[2.0], [1.0]])
    assert weighted_mean(values, mask, w)[0] == pytest.approx((2 * 1.0 + 3.0 + 5.0) / 4.0)
