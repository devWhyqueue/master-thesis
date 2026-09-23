"""Unit tests for exp-32's confusion-rate weighting and the reused worst.score search."""

from __future__ import annotations

from itertools import permutations

import numpy as np

from rate.intuition import _active_arcs
from rate.order import confusion_rates
from worst.score import score, search


def test_confusion_rates_hand_example() -> None:
    """Off-diagonal rate = count / full row total; an easy class with few errors gets small weight."""
    confusion = np.array(
        [
            [97, 2, 1],  # easy class: 100 total, 3 errors
            [10, 70, 20],  # confused class: 100 total, 30 errors
            [0, 0, 50],  # perfect class: 50 total, 0 errors
        ]
    )
    w = confusion_rates(confusion)
    np.testing.assert_allclose(w[0], [0.0, 0.02, 0.01])
    np.testing.assert_allclose(w[1], [0.10, 0.0, 0.20])
    np.testing.assert_allclose(w[2], [0.0, 0.0, 0.0])
    assert w.sum(axis=1).max() <= 1.0
    assert w[0].sum() < w[1].sum()  # easy class carries less total weight than the confused one


def test_search_matches_brute_force_over_all_permutations() -> None:
    """search(maximize=True/False) matches brute force over all 5! permutations.

    Reused from exp-31: the rate weighting only changes how w is built, not the score/search
    contract, so this is the same check against the same reused ``worst.score`` functions.
    """
    rng = np.random.default_rng(0)
    k = 5
    w = rng.random((k, k))
    np.fill_diagonal(w, 0.0)
    h = rng.random(k)
    z = np.sort(rng.random(k))[::-1]  # decreasing with rank, as the real z-profile is

    brute = [(score(np.array(p), w, h, z), p) for p in permutations(range(k))]
    best_max = max(s for s, _ in brute)
    best_min = min(s for s, _ in brute)

    found_max = search(w, h, z, maximize=True, seed=1, n_restarts=100)
    found_min = search(w, h, z, maximize=False, seed=1, n_restarts=100)

    assert score(found_max, w, h, z) == best_max
    assert score(found_min, w, h, z) == best_min


def test_active_arcs_keep_only_classes_below_their_partner() -> None:
    """An arc is drawn from a class's rank to its partner's rank only when the class sits lower."""
    order = ["a", "b", "c"]
    partner = {"a": "b", "b": "a", "c": "a"}
    assert _active_arcs(order, partner) == [(1, 0), (2, 0)]
