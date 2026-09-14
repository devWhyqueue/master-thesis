"""Unit tests for coverage-redundancy omega, Neff^omega, the ICC identity, and OLS."""

from __future__ import annotations

import numpy as np
import pytest

from breadth.surface import fit_ols

from redundancy.estimator import (
    _between_sum_of_squares,
    _weighted_sums,
    cell_effective_support,
    cluster_stats,
    weighted_icc,
)

from coverage_redundancy.analyze import classify
from coverage_redundancy.models import design
from coverage_redundancy.quantities import neff_omega, omega_similarity


@pytest.mark.parametrize(
    ("b_ci", "con_ci", "sel_ci", "expected"),
    [
        ((-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5), "coverage_and_similarity"),
        ((-0.5, 0.5), (1.5, 2.0), (-0.5, 0.5), "random_cohorts_only"),
        ((-0.5, 0.5), (-2.0, -1.5), (-0.5, 0.5), "random_cohorts_only"),
        ((1.5, 2.0), (1.5, 2.0), (1.5, 2.0), "breadth_beyond_both"),
        ((0.5, 1.5), (-0.5, 0.5), (-0.5, 0.5), "inconclusive"),  # b straddles
        ((-0.5, 0.5), (0.5, 1.5), (-0.5, 0.5), "inconclusive"),  # con straddles
        ((-0.5, 0.5), (-0.5, 0.5), (1.5, 2.0), "inconclusive"),  # con within, sel above
    ],
)
def test_classify_truth_table(b_ci, con_ci, sel_ci, expected):
    assert classify(b_ci, con_ci, sel_ci) == expected


def test_omega_identical_deviations_equals_norm_squared_over_tau2():
    """Two cohort patients sharing one deviation from mu: omega = ||d||^2 / tau^2."""
    means = {
        ("p1", "c"): np.array([2.0, 0.0]),
        ("p2", "c"): np.array([2.0, 0.0]),
        ("p3", "c"): np.array([0.0, 0.0]),
        ("p4", "c"): np.array([0.0, 0.0]),
    }
    eligible = ["p1", "p2", "p3", "p4"]
    tau2 = 2.0
    # mu = (1, 0); both cohort patients deviate by d = (1, 0).
    omega = omega_similarity(means, ["p1", "p2"], eligible, "c", tau2)
    assert omega == pytest.approx(1.0 / tau2)


def test_omega_orthogonal_equal_norm_is_zero():
    """Pairwise-orthogonal, equal-norm deviations contribute zero cross terms."""
    means = {
        ("p1", "c"): np.array([1.0, 0.0, 0.0]),
        ("p2", "c"): np.array([0.0, 1.0, 0.0]),
        ("p3", "c"): np.array([0.0, 0.0, 1.0]),
        ("p4", "c"): np.array([-1.0, -1.0, -1.0]),
    }
    eligible = ["p1", "p2", "p3", "p4"]  # mu = (0, 0, 0)
    omega = omega_similarity(means, ["p1", "p2", "p3"], eligible, "c", tau2=3.0)
    assert omega == pytest.approx(0.0, abs=1e-9)


def test_omega_matches_brute_force_double_sum():
    """The closed form matches a literal double sum over cohort patient pairs."""
    rng = np.random.default_rng(0)
    eligible = [f"p{i}" for i in range(6)]
    vectors = {p: rng.normal(size=4) for p in eligible}
    means = {(p, "c"): v for p, v in vectors.items()}
    patients = eligible[:4]
    mu = np.mean([vectors[p] for p in eligible], axis=0)
    tau2 = 1.7
    g = len(patients)
    total = 0.0
    for i in patients:
        for j in patients:
            if i == j:
                continue
            total += float((vectors[i] - mu) @ (vectors[j] - mu))
    expected = total / (g * (g - 1) * tau2)
    actual = omega_similarity(means, patients, eligible, "c", tau2)
    assert actual == pytest.approx(expected)


def test_neff_omega_zero_matches_cell_effective_support():
    """With omega = 0 for every class, Neff^omega reduces to Neff (model a)."""
    rho_bar = {"a": 0.3, "b": 0.6}
    omega = {"a": 0.0, "b": 0.0}
    g, m = 10, 8
    actual = neff_omega(g, m, rho_bar, omega, ["a", "b"])
    rho_arr = np.array([[rho_bar["a"], rho_bar["b"]]])
    expected = float(cell_effective_support(rho_arr, [(g, m)])[0, 0])
    assert actual == pytest.approx(expected)


def test_tau2_icc_identity_matches_weighted_icc():
    """(B - W) / (B + (m_tilde - 1) W), built from the same parts as tau^2, is the ICC."""
    rng = np.random.default_rng(1)
    case_ids = np.array(["a", "a", "a", "b", "b", "c", "c", "c", "c"])
    features = rng.normal(size=(len(case_ids), 3))
    stats = cluster_stats(features, case_ids)
    w = np.ones((1, len(stats.cases)))
    sums = _weighted_sums(stats, w)
    ssb = _between_sum_of_squares(stats, sums.wn, sums.total_n)
    between_ms = ssb / (sums.h - 1.0)
    within_ms = sums.ssw_weighted / (sums.total_n - sums.h)
    m_tilde = (sums.total_n - sums.sum_wn2 / sums.total_n) / (sums.h - 1.0)
    icc_from_parts = (between_ms - within_ms) / (between_ms + (m_tilde - 1.0) * within_ms)
    expected = weighted_icc(stats, w)
    assert icc_from_parts == pytest.approx(expected)


def test_split_intercept_ols_recovers_known_coefficients():
    """Noiseless data generated from a known split-intercept model is recovered exactly."""
    rng = np.random.default_rng(2)
    n_per_split = 15
    split_idx = np.repeat([0, 1, 2], n_per_split)
    x1 = rng.normal(size=len(split_idx))
    x2 = rng.normal(size=len(split_idx))
    true_theta = np.array([5.0, 1.0, -2.0, 0.5, 3.0])  # alpha0, d1, d2, beta, gamma

    x = design(split_idx, [x1, x2])
    x_matrix = np.column_stack([np.ones(len(split_idx)), x])
    y = x_matrix @ true_theta

    theta, res_std, r2 = fit_ols(x, y)
    assert theta == pytest.approx(true_theta, abs=1e-8)
    assert res_std == pytest.approx(0.0, abs=1e-6)
    assert r2 == pytest.approx(1.0)
