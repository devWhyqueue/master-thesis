"""Unit tests for exp-30's tail-order permutations and the piecewise recall model."""

from __future__ import annotations

import numpy as np

from permutation.model import Coefficients, fit_piecewise, predict_d
from permutation.order import order_perm

_ORDER = ["low", "mid", "high"]  # ascending pooled r1 recall


def test_easy_tail_places_highest_recall_class_last() -> None:
    """Easy-tail's last rank is the highest-recall class, in the given split's own class order."""
    split_names = ["mid", "high", "low"]
    perm = order_perm(_ORDER, split_names, easy=True)
    assert perm[-1] == split_names.index("high")
    assert perm[0] == split_names.index("low")


def test_hard_tail_is_the_reverse_of_easy_tail() -> None:
    """Hard-tail reverses easy-tail's rank assignment, so its last rank is the lowest-recall class."""
    split_names = ["low", "mid", "high"]
    easy = order_perm(_ORDER, split_names, easy=True)
    hard = order_perm(_ORDER, split_names, easy=False)
    np.testing.assert_array_equal(hard, easy[::-1])
    assert hard[-1] == split_names.index("low")


def test_order_perm_respects_split_local_name_order() -> None:
    """The permutation maps by class name, not by canonical position, into the split's own order."""
    perm = order_perm(_ORDER, ["high", "low", "mid"], easy=True)
    # easy-tail rank order is low, mid, high -> local indices 1, 2, 0
    np.testing.assert_array_equal(perm, [1, 2, 0])


def test_piecewise_ls_recovers_known_coefficients() -> None:
    """Noiseless synthetic data (1 replicate, uniform weights) recovers exact a/b+/b- per class."""
    a_true = np.array([1.0, -2.0])
    b_pos_true = np.array([3.0, 0.5])
    b_neg_true = np.array([0.2, 4.0])
    z = np.array([[-2.0, -1.0, 0.0, 1.0, 2.0], [-3.0, -1.0, 0.0, 2.0, 4.0]])  # (C, N)
    delta = (
        a_true[:, None]
        + b_pos_true[:, None] * np.clip(z, 0, None)
        + b_neg_true[:, None] * np.clip(z, None, 0)
    )
    z3, delta3 = z[:, :, None], delta[:, :, None]  # (C, N, R=1)
    weight = np.ones_like(z3)
    coefs = fit_piecewise(z3, delta3, weight)
    np.testing.assert_allclose(coefs.a[:, 0], a_true, atol=1e-8)
    np.testing.assert_allclose(coefs.b_pos[:, 0], b_pos_true, atol=1e-8)
    np.testing.assert_allclose(coefs.b_neg[:, 0], b_neg_true, atol=1e-8)


def test_predict_d_of_balanced_counts_equals_negative_mean_intercept() -> None:
    """At z = 0 (the balanced r1 count), predicted D collapses to -mean_c(a_c)."""
    coefs = Coefficients(
        a=np.array([[1.0], [-2.0], [0.5]]),
        b_pos=np.array([[3.0], [0.5], [1.0]]),
        b_neg=np.array([[0.2], [4.0], [1.0]]),
    )
    z_balanced = np.zeros(3)
    d = predict_d(coefs, z_balanced)
    np.testing.assert_allclose(d, [-np.mean(coefs.a[:, 0])])
