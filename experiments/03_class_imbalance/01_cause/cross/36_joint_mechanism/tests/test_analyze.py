"""Unit tests for the channel decomposition arithmetic and the simultaneous-CI construction."""

from __future__ import annotations

import numpy as np

from joint import ARMS, SETTINGS
from joint.analyze import combine, simultaneous_interval


def _toy_ba(seed: int = 0, r: int = 50) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        f"{setting}_{arm}": rng.uniform(40.0, 90.0, size=r)
        for setting in SETTINGS
        for arm in ARMS
    }


def test_combine_computes_damage_as_balanced_minus_each_arm():
    ba = _toy_ba()
    dists = combine(ba)
    for setting in SETTINGS:
        b, p, s, r = (ba[f"{setting}_{arm}"] for arm in ARMS)
        np.testing.assert_allclose(dists[f"D_P_{setting}"], b - p)
        np.testing.assert_allclose(dists[f"D_S_{setting}"], b - s)
        np.testing.assert_allclose(dists[f"D_R_{setting}"], b - r)


def test_combine_interaction_is_total_minus_prior_minus_support():
    ba = _toy_ba()
    dists = combine(ba)
    for setting in SETTINGS:
        expected = dists[f"D_R_{setting}"] - dists[f"D_P_{setting}"] - dists[f"D_S_{setting}"]
        np.testing.assert_allclose(dists[f"I_{setting}"], expected)


def test_simultaneous_interval_is_narrower_at_looser_bonferroni_budget():
    """More simultaneous estimates -> a tighter per-estimate alpha -> a wider interval."""
    rng = np.random.default_rng(1)
    dist = np.concatenate([[5.0], rng.normal(5.0, 1.0, size=2000)])
    two = simultaneous_interval(dist, n_estimates=2)
    one = simultaneous_interval(dist, n_estimates=1)
    assert two["point"] == one["point"] == 5.0
    assert (two["ci_upper"] - two["ci_lower"]) >= (one["ci_upper"] - one["ci_lower"])


def test_simultaneous_interval_point_is_the_observed_replicate_not_the_bootstrap_mean():
    dist = np.array([3.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    result = simultaneous_interval(dist)
    assert result["point"] == 3.5
