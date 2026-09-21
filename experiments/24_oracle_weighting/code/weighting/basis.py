"""Pure construction of the pool's own complement basis and the oracle-weighted expanded covariance."""

from __future__ import annotations

import numpy as np

from centre.pool import Pool

from spectrum.basis import oracle_eigvals

__all__ = ["complement_pool", "oracle_expanded"]

_RANK_TOL = 1e-10


def complement_pool(u_b: np.ndarray, pool: Pool) -> tuple[np.ndarray, np.ndarray]:
    """(k, d) eigenvectors and (k,) eigenvalues of the pool between-patient covariance projected off ``u_b``."""
    proj = pool.b_basis - (pool.b_basis @ u_b.T) @ u_b
    _, s, vt = np.linalg.svd(
        np.sqrt(pool.b_eigvals)[:, None] * proj, full_matrices=False
    )
    keep = s > _RANK_TOL * s[0]
    return vt[keep], s[keep] ** 2


def oracle_expanded(
    u_b: np.ndarray,
    lam_b: np.ndarray,
    v: np.ndarray,
    pool: Pool,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Basis and eigenvalues of the cohort block plus ``v`` weighted by ``weights`` (default: pool variance along ``v``)."""
    w = oracle_eigvals(v, pool) if weights is None else weights
    return np.vstack([u_b, v]), np.concatenate([lam_b, w])
