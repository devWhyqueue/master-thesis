"""Unit tests for the x^(alpha) centre-shift intervention and the separation index J."""

from __future__ import annotations

import numpy as np
import pytest

from separation.geometry import Centres, apply_intervention, separation_index


def _toy_centres(seed: int = 0, c: int = 4, d: int = 6) -> Centres:
    rng = np.random.default_rng(seed)
    means = rng.normal(size=(c, d))
    return Centres([f"c{i}" for i in range(c)], means, means.mean(axis=0), 1.0)


def test_apply_intervention_is_identity_at_alpha_one():
    """alpha = 1 leaves every row unchanged."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=(20, 6))
    y = np.arange(20) % 4
    centres = _toy_centres()
    np.testing.assert_array_equal(apply_intervention(x, y, centres, 1.0), x)


def test_apply_intervention_preserves_within_class_differences():
    """Any two rows of the same class keep their exact difference after the shift."""
    rng = np.random.default_rng(2)
    x = rng.normal(size=(20, 6))
    y = np.arange(20) % 4
    centres = _toy_centres()
    shifted = apply_intervention(x, y, centres, 2.5)
    same_class = np.flatnonzero(y == y[0])
    np.testing.assert_allclose(
        shifted[same_class] - shifted[same_class][0],
        x[same_class] - x[same_class][0],
        atol=1e-12,
    )


def test_apply_intervention_moves_class_mean_by_alpha_offset():
    """A class's mean moves by exactly (alpha - 1) times its centre's offset from the grand mean."""
    rng = np.random.default_rng(3)
    x = rng.normal(size=(400, 6))
    y = np.arange(400) % 4
    centres = _toy_centres()
    alpha = 3.0
    shifted = apply_intervention(x, y, centres, alpha)
    for c in range(4):
        expected_shift = (alpha - 1.0) * (centres.means[c] - centres.grand_mean)
        actual_shift = shifted[y == c].mean(axis=0) - x[y == c].mean(axis=0)
        np.testing.assert_allclose(actual_shift, expected_shift, atol=1e-10)


def test_separation_index_scales_with_within_class_rms():
    """Halving the within-class RMS doubles J; J is invariant to a uniform centre translation."""
    means = np.array([[0.0, 0.0], [4.0, 0.0], [0.0, 4.0]])
    centres = Centres(["a", "b", "c"], means, means.mean(axis=0), within_rms=2.0)
    j = separation_index(centres)
    assert j == pytest.approx(2.0)  # nearest-centre distance 4.0 / within_rms 2.0
    halved = centres._replace(within_rms=1.0)
    assert separation_index(halved) == pytest.approx(2.0 * j)
    translated = centres._replace(means=means + 100.0)
    assert separation_index(translated) == pytest.approx(j)
