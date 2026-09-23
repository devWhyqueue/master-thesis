"""Unit tests for exp-34's arm sourcing, alpha keys, and shard-index decoding."""

from __future__ import annotations

import pytest

from separation import ARMS, FIT_SOURCE, NEW_SETTINGS_BY_DATASET, main_settings, pilot_settings
from separation.fit import (
    _ALPHA_KEY,
    decode_main_shard_index,
    decode_pilot_shard_index,
    main_shard_count,
    pilot_shard_count,
)


def test_fit_source_covers_every_arm():
    """Every arm has a (data_arm, prior_arm) source; B and R are unweighted."""
    assert set(FIT_SOURCE) == set(ARMS)
    assert FIT_SOURCE["B"] == ("r1", None)
    assert FIT_SOURCE["R"] == ("r100", None)
    assert FIT_SOURCE["P"] == ("r1", "r100")
    assert FIT_SOURCE["S"] == ("r100", "r1")


def test_alpha_key_covers_every_pilot_and_main_setting():
    """Every setting fit in either phase, for either dataset, has an alpha key entry."""
    settings = set()
    for dataset in ("bracs", "tcga_ut"):
        settings |= set(pilot_settings(dataset)) | set(main_settings(dataset))
    assert settings == set(_ALPHA_KEY)


@pytest.mark.parametrize("dataset", ["bracs", "tcga_ut"])
def test_pilot_shard_indices_cover_every_setting_and_split_once(dataset: str):
    count = pilot_shard_count(dataset)
    decoded = {decode_pilot_shard_index(dataset, i) for i in range(count)}
    settings = pilot_settings(dataset)
    assert decoded == {(setting, s, 0) for setting in settings for s in range(3)}
    with pytest.raises(ValueError):
        decode_pilot_shard_index(dataset, count)


@pytest.mark.parametrize("dataset", ["bracs", "tcga_ut"])
def test_main_shard_indices_cover_every_setting_split_and_draw_once(dataset: str):
    count = main_shard_count(dataset)
    decoded = {decode_main_shard_index(dataset, i) for i in range(count)}
    settings = main_settings(dataset)
    assert decoded == {
        (setting, s, d) for setting in settings for s in range(3) for d in (1, 2, 3, 4)
    }
    with pytest.raises(ValueError):
        decode_main_shard_index(dataset, count)


def test_pilot_and_main_condition_counts_match_the_plan():
    """36 new pilot conditions + 12 replay conditions; 144 main conditions (PLAN.md totals)."""
    new_pilot = sum(
        3 * len(NEW_SETTINGS_BY_DATASET[d]) for d in ("bracs", "tcga_ut")
    ) * len(ARMS)
    replay_pilot = 3 * len(ARMS)
    main = sum(main_shard_count(d) for d in ("bracs", "tcga_ut")) * len(ARMS)
    assert new_pilot == 36
    assert replay_pilot == 12
    assert main == 144
    assert new_pilot + replay_pilot + main == 192
