"""Unit tests for exp-31's directed pair-separation score, its local search, and rank orders."""

from __future__ import annotations

from itertools import combinations, permutations

import numpy as np

from worst.order import order_perm
from worst.score import score, search


def test_score_matches_hand_computed_three_class_value() -> None:
    """S on a hand-picked 3-class instance matches the value worked out by hand."""
    perm = np.array([0, 1, 2])  # rank 0 (head) = class 0, rank 2 (tail) = class 2
    z = np.array([1.0, 0.0, -1.0])
    w = np.array(
        [
            [0.0, 0.5, 0.5],
            [1.0, 0.0, 0.0],
            [0.3, 0.7, 0.0],
        ]
    )
    h = np.array([10.0, 20.0, 30.0])
    # c=1: w[1,0]*h1*max(z0-z1,0) = 1.0*20*1 = 20
    # c=2: w[2,0]*h2*max(z0-z2,0) + w[2,1]*h2*max(z1-z2,0) = 0.3*30*2 + 0.7*30*1 = 39
    assert score(perm, w, h, z) == 59.0


def test_search_matches_brute_force_over_all_permutations() -> None:
    """search(maximize=True/False) matches brute force over all 5! permutations."""
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


def _undirected_gap_sum(rank_of: dict[str, int]) -> int:
    return sum(abs(rank_of[a] - rank_of[b]) for a, b in combinations(rank_of, 2))


def test_flip_is_reversed_sep_with_identical_undirected_rank_gaps() -> None:
    """flip == reversed(sep); the undirected rank-gap sum is identical for both."""
    sep = ["a", "b", "c", "d", "e"]
    flip = list(reversed(sep))
    assert flip == sep[::-1]

    rank_of_sep = {c: i for i, c in enumerate(sep)}
    rank_of_flip = {c: i for i, c in enumerate(flip)}
    assert _undirected_gap_sum(rank_of_sep) == _undirected_gap_sum(rank_of_flip)


def test_order_perm_respects_split_local_name_order() -> None:
    """The permutation maps by class name, not by canonical position, into the split's own order."""
    perm = order_perm(["low", "mid", "high"], ["high", "low", "mid"])
    # rank order low, mid, high -> local indices 1, 2, 0
    np.testing.assert_array_equal(perm, [1, 2, 0])
