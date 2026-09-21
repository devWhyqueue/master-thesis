"""Pure constructions of the within-patient complement and the expanded whitening covariance."""

from __future__ import annotations

import numpy as np

from centre.arms import between_patient_eigenbasis
from centre.cohort import TrainingTable
from centre.pool import Pool

from spectrum.basis import pool_variance_captured

__all__ = [
    "complement_within",
    "random_complement",
    "expanded",
    "captured_top_k",
    "complement_oracle_share",
]


def complement_within(
    table: TrainingTable, n_classes: int, g: int, u_b: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """(k, d) eigenvectors and (k,) eigenvalues of the within-patient patch covariance projected off ``u_b``."""
    m = len(table.x) // (n_classes * g)
    x = table.x.reshape(n_classes * g, m, -1)
    dev = (x - x.mean(axis=1, keepdims=True)).reshape(len(table.x), -1)
    dev -= (dev @ u_b.T) @ u_b
    return between_patient_eigenbasis(dev, len(dev) - n_classes * g)


def random_complement(u_b: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """(k, d) random orthonormal rows orthogonal to the rows of ``u_b``."""
    z = rng.standard_normal((k, u_b.shape[1]))
    z -= (z @ u_b.T) @ u_b
    return np.linalg.qr(z.T)[0].T


def expanded(
    u_b: np.ndarray, lam_b: np.ndarray, v: np.ndarray, mu: np.ndarray, s: float
) -> tuple[np.ndarray, np.ndarray, float]:
    """Basis, eigenvalues, and tau of ``Sigma_B + s * tau * V diag(mu) V'`` with added trace ``s * sum(lam_b)``."""
    tau = float(lam_b.sum() / mu.sum())
    return np.vstack([u_b, v]), np.concatenate([lam_b, s * tau * mu]), tau


def captured_top_k(basis: np.ndarray, eigvals: np.ndarray, pool: Pool, k: int) -> float:
    """Pool between-patient variance share in the span of the ``k`` leading eigenvectors."""
    top = np.argsort(-eigvals)[:k]
    return pool_variance_captured(basis[top], pool)


def complement_oracle_share(u_b: np.ndarray, pool: Pool, k: int) -> float:
    """Pool variance share of the top ``k`` eigenvectors of the pool covariance projected off ``u_b``."""
    proj = pool.b_basis - (pool.b_basis @ u_b.T) @ u_b
    s = np.linalg.svd(np.sqrt(pool.b_eigvals)[:, None] * proj, compute_uv=False)
    return float((s[:k] ** 2).sum() / pool.b_eigvals.sum())
