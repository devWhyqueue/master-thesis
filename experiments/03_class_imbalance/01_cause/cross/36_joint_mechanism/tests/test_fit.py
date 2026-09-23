"""Unit tests for exp-36's shard-index decoding: one shard is one (split, draw), shared by every
setting/arm/control, so allocation and priors stay unchanged across a shard's own work.
"""

from __future__ import annotations

import pytest

from joint import ARMS, FIT_SOURCE, MAIN_DRAWS, N_SPLITS, PILOT_DRAWS
from joint.fitting.fit import decode_shard_index, main_shard_count, pilot_shard_count


def test_fit_source_covers_every_arm():
    """B and R are unweighted; P and S reweight toward the opposite prior (exp-27/28's scheme)."""
    assert set(FIT_SOURCE) == set(ARMS)
    assert FIT_SOURCE["B"] == ("r1", None)
    assert FIT_SOURCE["R"] == ("r100", None)
    assert FIT_SOURCE["P"] == ("r1", "r100")
    assert FIT_SOURCE["S"] == ("r100", "r1")


def test_pilot_shard_count_is_splits_times_pilot_draws():
    assert pilot_shard_count() == N_SPLITS * len(PILOT_DRAWS) == 6


def test_main_shard_count_is_splits_times_main_draws():
    assert main_shard_count() == N_SPLITS * len(MAIN_DRAWS) == 24


def test_pilot_shard_indices_cover_every_split_and_draw_once():
    count = pilot_shard_count()
    decoded = {decode_shard_index("pilot", i) for i in range(count)}
    assert decoded == {(s, d) for s in range(N_SPLITS) for d in PILOT_DRAWS}
    with pytest.raises(ValueError):
        decode_shard_index("pilot", count)


def test_main_shard_indices_cover_every_split_and_draw_once():
    count = main_shard_count()
    decoded = {decode_shard_index("main", i) for i in range(count)}
    assert decoded == {(s, d) for s in range(N_SPLITS) for d in MAIN_DRAWS}
    with pytest.raises(ValueError):
        decode_shard_index("main", count)


def test_pilot_and_main_arm_evaluation_counts_match_the_plan():
    """192 core + 24 wrong-direction + 84 fixed-lambda = 300 pilot; 4x that scale for main
    (PLAN.md "Execution and genuine precheck"), pooled over both datasets."""
    n_datasets = 2
    settings = 4
    core_per_shard = settings * len(ARMS)  # 16
    wrong_per_shard = 2  # joint S/R
    fixed_per_shard = 3 + 4  # native P/S/R + joint B/P/S/R
    per_shard = core_per_shard + wrong_per_shard + fixed_per_shard
    assert per_shard == 25
    pilot_shards = n_datasets * pilot_shard_count()
    main_shards = n_datasets * main_shard_count()
    assert per_shard * pilot_shards == 300
    assert per_shard * main_shards == 1200
    assert core_per_shard * pilot_shards == 192
    assert wrong_per_shard * pilot_shards == 24
    assert fixed_per_shard * pilot_shards == 84
