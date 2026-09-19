"""Pure spectrum and subspace constructions for the cohort's between-patient covariance."""

from __future__ import annotations

import numpy as np

from centre.arms import between_patient_eigenbasis
from centre.cohort import TrainingTable
from centre.pool import Pool

__all__ = [
    "spectrum_eigvals",
    "oracle_eigvals",
    "within_patient_basis",
    "oracle_slope",
    "pool_variance_captured",
    "subspace_overlap",
]


def spectrum_eigvals(eigvals: np.ndarray, beta: float) -> np.ndarray:
    """Power-compressed spectrum ``eigvals ** beta``; beta = 1 keeps it, beta = 0 flattens it."""
    return eigvals**beta


def oracle_eigvals(basis: np.ndarray, pool: Pool) -> np.ndarray:
    """(k,) pool between-patient variance along each cohort direction, ``u_j' Sigma_pool u_j``."""
    return ((basis @ pool.b_basis.T) ** 2) @ pool.b_eigvals


def within_patient_basis(
    table: TrainingTable, n_classes: int, g: int, k: int
) -> np.ndarray:
    """(<=k, d) top eigenvectors of the cohort's pooled within-patient patch covariance."""
    m = len(table.x) // (n_classes * g)
    x = table.x.reshape(n_classes * g, m, -1)
    dev = (x - x.mean(axis=1, keepdims=True)).reshape(len(table.x), -1)
    basis, _ = between_patient_eigenbasis(dev, len(dev) - n_classes * g)
    return basis[:k]


def oracle_slope(eigvals: np.ndarray, oracle: np.ndarray) -> float:
    """OLS slope of log oracle variance on log cohort eigenvalue: the empirical oracle beta."""
    return float(np.polyfit(np.log(eigvals), np.log(oracle), 1)[0])


def pool_variance_captured(basis: np.ndarray, pool: Pool) -> float:
    """Share of the pool's between-patient variance lying in span(``basis``)."""
    return float(oracle_eigvals(basis, pool).sum() / pool.b_eigvals.sum())


def subspace_overlap(basis_a: np.ndarray, basis_b: np.ndarray) -> float:
    """``||A B'||_F^2 / k`` for orthonormal rows: 1 for equal subspaces, ~k/d for random ones."""
    return float(((basis_a @ basis_b.T) ** 2).sum() / len(basis_a))
