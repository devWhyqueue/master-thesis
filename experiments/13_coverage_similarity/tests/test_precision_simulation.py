"""Regression checks for the training and patient components of simulated studies."""

from __future__ import annotations

import numpy as np
import pytest

from similarity.simulate import _one_study


def test_draw_only_study_counts_training_fluctuation_once() -> None:
    """With identical patient replicates, the point is the sampled draw mean."""
    points = np.array([[-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
    template = np.repeat(points[:, :, None], 31, axis=2)
    seed, draws = 12, 20
    rng = np.random.default_rng(seed)
    counts = [rng.multinomial(draws, np.full(3, 1 / 3)) for _ in range(2)]
    expected = np.mean([count @ point / draws for count, point in zip(counts, points)])
    assert expected != 0
    actual = _one_study(np.random.default_rng(seed), {"only": template}, draws, 31)
    assert actual["only"][0] == pytest.approx(expected)


def test_mixed_study_adds_patient_shift_once_and_keeps_interval_width() -> None:
    """The same patient shift moves the point and all replicate draws together."""
    points = np.array([[-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
    patient_shift = np.linspace(-0.6, 0.6, 30)
    shifts = np.concatenate([[0.0], patient_shift])
    draw_only = np.repeat(points[:, :, None], 31, axis=2)
    mixed = draw_only + shifts
    seed, draws = 12, 20
    rng = np.random.default_rng(seed)
    counts = [rng.multinomial(draws, np.full(3, 1 / 3)) for _ in range(2)]
    world_column = int(rng.integers(1, 31))
    expected_point = np.mean(
        [count @ point / draws for count, point in zip(counts, points)]
    )
    actual = _one_study(np.random.default_rng(seed), {"only": mixed}, draws, 31)["only"]
    baseline = _one_study(np.random.default_rng(seed), {"only": draw_only}, draws, 31)[
        "only"
    ]
    assert actual[0] == pytest.approx(expected_point + shifts[world_column])
    np.testing.assert_allclose(actual - baseline, shifts + shifts[world_column])


def test_patient_only_uncertainty_persists_as_draw_count_increases() -> None:
    """Increasing training draws cannot reduce uncertainty from a fixed test set."""
    patient_shift = np.array([0.0, -0.6, -0.3, 0.3, 0.6])
    template = np.tile(patient_shift, (2, 3, 1))
    for draws in (20, 60, 600):
        actual = _one_study(np.random.default_rng(12), {"only": template}, draws, 5)[
            "only"
        ]
        np.testing.assert_allclose(actual - actual[0], patient_shift)
        assert np.any(np.isclose(actual[0], patient_shift[1:]))
