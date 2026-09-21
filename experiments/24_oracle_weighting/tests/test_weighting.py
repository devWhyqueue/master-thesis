"""Unit tests for the oracle-weighted complement constructions."""

from __future__ import annotations

import numpy as np

from centre.cohort import TrainingTable
from centre.pool import Pool

from span.basis import complement_within, random_complement
from span.fit import Context
from spectrum.basis import oracle_eigvals

from weighting import ARMS
from weighting.basis import complement_pool, oracle_expanded
from weighting.fit import _arm_pool


def _setup(seed: int = 0, pool_in_u: bool = False):
    """Tiny cohort (2 classes, 3 patients, 8 patches, d = 12) with a pool of rank 5."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((2 * 3 * 8, 12))
    table = TrainingTable(x, np.zeros(len(x), int))
    u_b = np.linalg.qr(rng.standard_normal((12, 3)))[0].T
    lam_b = np.array([3.0, 2.0, 1.0])
    v, mu = complement_within(table, 2, 3, u_b)
    b_basis = u_b[:2] if pool_in_u else np.linalg.qr(rng.standard_normal((12, 5)))[0].T
    b_eig = np.array([2.0, 1.0]) if pool_in_u else np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    pool = Pool([], np.empty(0), np.empty(0), b_basis, b_eig, np.empty(0), np.empty(0))
    return rng, u_b, lam_b, v, mu, pool


def test_oracle_weights_equal_pool_variance_along_v():
    """Added eigenvalues are diag(V Sigma_pool V')."""
    _, u_b, lam_b, v, _, pool = _setup()
    _, eigvals = oracle_expanded(u_b, lam_b, v, pool)
    sigma = pool.b_basis.T @ np.diag(pool.b_eigvals) @ pool.b_basis
    np.testing.assert_allclose(eigvals[len(lam_b) :], np.diag(v @ sigma @ v.T))


def test_complement_pool_orthonormal_orthogonal_and_variance_preserving():
    """Rows are orthonormal, orthogonal to U, and eigenvalues sum to pool variance outside span(U)."""
    _, u_b, _, _, _, pool = _setup()
    basis, eig = complement_pool(u_b, pool)
    np.testing.assert_allclose(basis @ basis.T, np.eye(len(basis)), atol=1e-10)
    np.testing.assert_allclose(basis @ u_b.T, 0.0, atol=1e-10)
    inside = oracle_eigvals(u_b, pool).sum()
    assert np.isclose(eig.sum(), pool.b_eigvals.sum() - inside)


def test_complement_pool_vanishes_when_pool_in_span_u():
    """A pool inside span(U) leaves no complement variance."""
    _, u_b, _, _, _, pool = _setup(pool_in_u=True)
    _, eig = complement_pool(u_b, pool)
    assert eig.sum() < 1e-8 or len(eig) == 0


def test_kappa_anchor_keeps_cohort_block_scale():
    """factor * anchor / mean(eigvals) equals factor / mean(lam_b) for every arm family."""
    rng, u_b, lam_b, v, mu, pool = _setup()
    ctx: Context = (u_b, lam_b, v, mu, random_complement(u_b, len(mu), rng))
    for family in ("Wo", "Ro", "Po"):
        arm_pool, _, anchor = _arm_pool(family, ctx, pool)
        kappa = 10.0 * anchor / arm_pool.b_eigvals.mean()
        assert np.isclose(kappa, 10.0 / lam_b.mean())
    assert ARMS == ("Wo5", "Ro5", "Po5")
