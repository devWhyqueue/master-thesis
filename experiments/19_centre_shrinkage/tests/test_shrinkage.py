"""Unit tests for James-Stein centre shrinkage, its weight formula, and derived gains/shares."""

from __future__ import annotations

import numpy as np
import pytest

from centre import PATIENT_COUNTS
from centre.cohort import TrainingTable
from centre.fit import split_arm

from shrinkage import ARMS, REUSED_ARMS
from shrinkage.analyze import derived
from shrinkage.centres import js_weights, patient_means, shrunk_centres


def test_shrunk_centres_have_smaller_total_squared_error_than_raw_means():
    """Averaged over draws, shrunk centres are closer to the true centres than the raw means."""
    rng = np.random.default_rng(11)
    n_classes, g, m, d = 6, 8, 3, 5
    true_centres = rng.normal(scale=0.3, size=(n_classes, d))
    raw_error = shrunk_error = 0.0
    for _ in range(200):
        sampled = true_centres[:, None, :] + rng.normal(size=(n_classes, g, d))
        x = np.repeat(sampled, m, axis=1).reshape(n_classes * g * m, d)
        table = TrainingTable(x, np.repeat(np.arange(n_classes), g * m))
        means = patient_means(table, n_classes, g)
        raw_error += ((means.mean(axis=1) - true_centres) ** 2).sum()
        shrunk_error += ((shrunk_centres(means, 1.0) - true_centres) ** 2).sum()
    assert shrunk_error < raw_error


def test_js_weights_edge_cases():
    """Zero between-patient variance gives w = 0; noise far larger than spread gives w = 1."""
    n_classes, g, d = 4, 4, 6
    centres = np.arange(n_classes * d, dtype=float).reshape(n_classes, d)
    zero_var = np.repeat(centres[:, None, :], g, axis=1)
    np.testing.assert_allclose(js_weights(zero_var), 0.0)

    # Symmetric +/- deviations cancel exactly, so the class mean stays pinned at
    # ``tiny_centres[c]`` regardless of how large the within-class spread is.
    rng = np.random.default_rng(12)
    tiny_centres = rng.normal(scale=1e-6, size=(n_classes, d))
    half = rng.normal(scale=10.0, size=(n_classes, g // 2, d))
    noisy = tiny_centres[:, None, :] + np.concatenate([half, -half], axis=1)
    assert np.all(js_weights(noisy) == 1.0)


def test_derived_on_toy_arm_gaps_gives_hand_computed_gains_recovery_and_shares():
    """Gains, recovery fractions, and gap shares match hand-computed values on toy arm distributions."""
    base = {
        "R": (60.0, 66.0, 70.0),
        "C": (70.0, 73.0, 75.0),
        "S": (62.0, 67.0, 70.5),
        "St": (63.0, 67.5, 70.5),
    }
    arm = {
        f"{f}{g}": np.array([v])
        for f, vals in base.items()
        for g, v in zip(PATIENT_COUNTS, vals)
    }

    out = derived(arm)

    assert out["S5_minus_R5"][0] == pytest.approx(2.0)
    assert out["St5_minus_R5"][0] == pytest.approx(3.0)
    assert out["recovery_S_5"][0] == pytest.approx(0.2)
    assert out["recovery_S_10"][0] == pytest.approx(1.0 / 7.0)
    assert out["gap_S_5_to_10"][0] == pytest.approx(5.0)
    assert out["share_S_5_to_10"][0] == pytest.approx(1.0 / 6.0)


def test_arm_names_round_trip_through_split_arm():
    """ARMS and REUSED_ARMS decode to their own (family, G) via centre's split_arm."""
    for arm in (*ARMS, *REUSED_ARMS):
        family, g = split_arm(arm)
        assert g in PATIENT_COUNTS
        assert arm == f"{family}{g}"
    assert split_arm("St10") == ("St", 10)
