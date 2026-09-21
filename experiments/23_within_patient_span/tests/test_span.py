"""Unit tests for the expanded whitening covariance and its random control."""

from __future__ import annotations

import numpy as np

from centre.arms import whiten
from centre.cohort import TrainingTable

from span import ARMS
from span.basis import complement_within, expanded, random_complement
from span.fit import Context, _arm_pool
from centre.pool import Pool


def _setup(seed: int = 0):
    """Cohort basis and complement of a tiny (2 classes, 3 patients, 8 patches, d = 12) table."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((2 * 3 * 8, 12))
    table = TrainingTable(x, np.zeros(len(x), int))
    u_b = np.linalg.qr(rng.standard_normal((12, 3)))[0].T
    lam_b = np.array([3.0, 2.0, 1.0])
    v, mu = complement_within(table, 2, 3, u_b)
    return rng, u_b, lam_b, v, mu


def test_s_zero_is_rwc_whitening():
    """s = 0 leaves the added directions unscaled, so whitening equals the cohort-only whitening."""
    rng, u_b, lam_b, v, mu = _setup()
    basis, eigvals, _ = expanded(u_b, lam_b, v, mu, 0.0)
    x = rng.standard_normal((5, 12))
    np.testing.assert_allclose(
        whiten(x, basis, eigvals, 0.7), whiten(x, u_b, lam_b, 0.7), atol=1e-12
    )


def test_expanded_basis_is_orthonormal_and_complementary():
    """Rows stay orthonormal and V is orthogonal to U_B."""
    _, u_b, lam_b, v, mu = _setup()
    basis, _, _ = expanded(u_b, lam_b, v, mu, 1.0)
    np.testing.assert_allclose(basis @ basis.T, np.eye(len(basis)), atol=1e-10)
    np.testing.assert_allclose(v @ u_b.T, 0.0, atol=1e-10)


def test_tau_sets_added_trace_to_s_times_between_trace():
    """The added eigenvalues sum to s * sum(lam_b)."""
    _, u_b, lam_b, v, mu = _setup()
    _, eigvals, _ = expanded(u_b, lam_b, v, mu, 0.3)
    assert np.isclose(eigvals[len(lam_b) :].sum(), 0.3 * lam_b.sum())


def test_random_control_stays_in_complement_and_keeps_mu():
    """Random rows are orthonormal, orthogonal to U_B, and receive the same eigenvalues."""
    rng, u_b, lam_b, v, mu = _setup()
    rand = random_complement(u_b, len(mu), rng)
    np.testing.assert_allclose(rand @ rand.T, np.eye(len(mu)), atol=1e-10)
    np.testing.assert_allclose(rand @ u_b.T, 0.0, atol=1e-10)
    _, eig_w, _ = expanded(u_b, lam_b, v, mu, 1.0)
    _, eig_r, _ = expanded(u_b, lam_b, rand, mu, 1.0)
    np.testing.assert_allclose(eig_w, eig_r)


def test_kappa_anchor_keeps_cohort_block_scale():
    """factor * anchor / mean(eigvals) equals factor / mean(lam_b) for every s."""
    rng, u_b, lam_b, v, mu = _setup()
    b_basis = np.linalg.qr(rng.standard_normal((12, 5)))[0].T
    pool = Pool([], np.empty(0), np.empty(0), b_basis, np.ones(5), np.empty(0), np.empty(0))
    ctx: Context = (u_b, lam_b, v, mu, random_complement(u_b, len(mu), rng))
    for family in ("Wa", "Wd", "Rc"):
        arm_pool, _, anchor = _arm_pool(family, ctx, pool)
        kappa = 10.0 * anchor / arm_pool.b_eigvals.mean()
        assert np.isclose(kappa, 10.0 / lam_b.mean())
    assert "Wt5" in ARMS and "Rt20" in ARMS
