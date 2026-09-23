"""Unit tests for the centre-correction target: translation identities and wrong-direction control."""

from __future__ import annotations

import numpy as np

from separation.geometry import Centres
from joint.target import move_to_target, move_to_target_negated, shift_target


def _toy_data(seed: int = 0, c: int = 4, n_per_class: int = 10, d: int = 6):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(c * n_per_class, d))
    y = np.repeat(np.arange(c), n_per_class)
    return x, y


def test_move_to_target_hits_the_target_exactly():
    """Every class's mean lands exactly on its target row."""
    x, y = _toy_data()
    target = np.random.default_rng(1).normal(size=(4, 6)) * 3.0
    moved = move_to_target(x, y, target)
    for c in range(4):
        np.testing.assert_allclose(moved[y == c].mean(axis=0), target[c], atol=1e-8)


def test_move_to_target_preserves_within_class_residuals():
    """Any two rows of the same class keep their exact difference after translation."""
    x, y = _toy_data()
    target = np.random.default_rng(2).normal(size=(4, 6))
    moved = move_to_target(x, y, target)
    same_class = np.flatnonzero(y == y[0])
    np.testing.assert_allclose(
        moved[same_class] - moved[same_class][0],
        x[same_class] - x[same_class][0],
        atol=1e-10,
    )


def test_move_to_target_is_identity_when_target_equals_current_means():
    """Translating to the class's own current mean changes nothing."""
    x, y = _toy_data()
    from centre.arms import class_means

    target = class_means(x, y, 4)
    moved = move_to_target(x, y, target)
    np.testing.assert_allclose(moved, x, atol=1e-8)


def test_move_to_target_negated_moves_the_opposite_way_by_the_same_magnitude():
    """The wrong-direction control applies the mirrored offset: same magnitude, opposite sign."""
    x, y = _toy_data()
    target = np.random.default_rng(3).normal(size=(4, 6))
    correct = move_to_target(x, y, target)
    wrong = move_to_target_negated(x, y, target)
    for c in range(4):
        correct_c = correct[y == c].mean(axis=0) - x[y == c].mean(axis=0)
        wrong_c = wrong[y == c].mean(axis=0) - x[y == c].mean(axis=0)
        np.testing.assert_allclose(wrong_c, -correct_c, atol=1e-8)


def test_shift_target_is_identity_at_alpha_one():
    """alpha = 1 leaves the native target unchanged."""
    means = np.random.default_rng(4).normal(size=(4, 6))
    centres = Centres(["a", "b", "c", "d"], means, means.mean(axis=0), 1.0)
    target = np.random.default_rng(5).normal(size=(4, 6))
    np.testing.assert_array_equal(shift_target(target, centres, 1.0), target)


def test_shift_target_applies_the_same_offset_as_the_separation_shift():
    """shift_target adds exactly (alpha - 1)(mu_c - mu_bar), matching apply_intervention's shift."""
    means = np.random.default_rng(6).normal(size=(4, 6))
    grand_mean = means.mean(axis=0)
    centres = Centres(["a", "b", "c", "d"], means, grand_mean, 1.0)
    target = np.random.default_rng(7).normal(size=(4, 6))
    alpha = 2.5
    shifted = shift_target(target, centres, alpha)
    expected = target + (alpha - 1.0) * (means - grand_mean)
    np.testing.assert_allclose(shifted, expected, atol=1e-12)
