"""Per (split, class) hull geometry and batched cohort quantities: coverage, hull residual, similarity."""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
import pandas as pd

from neighbours.coverage import _patients_of

from similarity.geometry import ClassGeometry, SplitInputs, build_split_geometry

__all__ = [
    "HullGeometry",
    "CohortValues",
    "build_hull_geometry",
    "hull_residuals",
    "pool_values",
    "cohort_values",
]

_EIG_REL_TOL = 1e-9
_CHUNK = 256


class HullGeometry(NamedTuple):
    """One (split, class)'s exp-14 pool geometry plus raw patient-mean Gram matrices."""

    base: ClassGeometry
    k_pool: np.ndarray  # (P, P) Gram of raw pool patient means
    k_val: np.ndarray  # (V, P) cross Gram, validation x pool
    n_val: np.ndarray  # (V,) squared norms of validation patient means


class CohortValues(NamedTuple):
    """Coverage distance, hull residual, and similarity of a batch of cohorts."""

    r_train: np.ndarray
    h_train: np.ndarray
    r_val: np.ndarray
    h_val: np.ndarray
    omega: np.ndarray


def build_hull_geometry(
    config: dict[str, Any],
    split_idx: int,
    full_df: pd.DataFrame,
    train_df: pd.DataFrame,
    shared: SplitInputs,
) -> dict[str, HullGeometry]:
    """Every class's hull geometry for one split, on top of exp-14's pool geometry."""
    out: dict[str, HullGeometry] = {}
    base_by_class = build_split_geometry(config, split_idx, full_df, train_df, shared)
    for c_name, base in base_by_class.items():
        val_patients = _patients_of(full_df, "validation", c_name)
        pool = np.stack([shared.means[(p, c_name)] for p in base.pool])
        val = np.stack([shared.means[(p, c_name)] for p in val_patients])
        pool, val = pool.astype(np.float64), val.astype(np.float64)
        out[c_name] = HullGeometry(
            base, pool @ pool.T, val @ pool.T, np.einsum("ij,ij->i", val, val)
        )
    return out


def _hull_chunk(
    idx: np.ndarray, k_pool: np.ndarray, k_ref: np.ndarray, n_ref: np.ndarray
) -> np.ndarray:
    k_ss = k_pool[idx[:, :, None], idx[:, None, :]]
    k_bar = k_ss.mean(axis=2)
    k_all = k_ss.mean(axis=(1, 2))
    cross = np.transpose(k_ref[:, idx], (1, 0, 2))
    c_bar = cross.mean(axis=2)
    gram = k_ss - k_bar[:, :, None] - k_bar[:, None, :] + k_all[:, None, None]
    proj = cross - c_bar[:, :, None] - k_bar[:, None, :] + k_all[:, None, None]
    sq_norm = n_ref[None, :] - 2.0 * c_bar + k_all[:, None]
    eigval, eigvec = np.linalg.eigh(gram)
    keep = eigval > _EIG_REL_TOL * np.abs(eigval).max(axis=1, keepdims=True)
    inv = np.where(keep, 1.0 / np.where(keep, eigval, 1.0), 0.0)
    explained = ((proj @ eigvec) ** 2 * inv[:, None, :]).sum(axis=2)
    return (sq_norm - explained) / sq_norm


def hull_residuals(
    idx: np.ndarray, k_pool: np.ndarray, k_ref: np.ndarray, n_ref: np.ndarray
) -> np.ndarray:
    """(N, R) share of each reference patient's deviation from the cohort mean outside the cohort's span.

    ``idx`` is (N, G) pool indices; ``k_ref`` is the (R, P) reference x pool
    Gram and ``n_ref`` the (R,) reference squared norms. Everything is
    expressed through Gram matrices, so no 2,560-dimensional projection is formed.
    """
    return np.concatenate(
        [
            _hull_chunk(idx[s : s + _CHUNK], k_pool, k_ref, n_ref)
            for s in range(0, len(idx), _CHUNK)
        ]
    )


def _leave_out_mean(values: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """(N,) mean over pool reference rows, excluding each cohort's own members."""
    mask = np.ones(values.shape, dtype=bool)
    np.put_along_axis(mask, idx, False, axis=1)
    return (values * mask).sum(axis=1) / mask.sum(axis=1)


def _coverage_rows(d_ref_pool: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return np.transpose(d_ref_pool[:, idx].min(axis=2))


def _omega(base: ClassGeometry, idx: np.ndarray) -> np.ndarray:
    g = idx.shape[1]
    if base.tau2 <= 0:
        return np.zeros(len(idx))
    sub = base.gram[idx[:, :, None], idx[:, None, :]]
    off_diag = sub.sum(axis=(1, 2)) - np.trace(sub, axis1=1, axis2=2)
    return off_diag / (g * (g - 1) * base.tau2)


def pool_values(
    geo: HullGeometry, idx: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Leave-out pool coverage distance, pool hull residual, and similarity of (N, G) cohorts."""
    n_pool = np.diag(geo.k_pool)
    r_rows = _coverage_rows(geo.base.d_pool, idx)
    h_rows = hull_residuals(idx, geo.k_pool, geo.k_pool, n_pool)
    return (
        _leave_out_mean(r_rows, idx),
        _leave_out_mean(h_rows, idx),
        _omega(geo.base, idx),
    )


def cohort_values(geo: HullGeometry, idx: np.ndarray) -> CohortValues:
    """Pool (search) and validation (analysis) quantities of (N, G) cohorts."""
    r_train, h_train, omega = pool_values(geo, idx)
    r_val = _coverage_rows(geo.base.d_val, idx).mean(axis=1)
    h_val = hull_residuals(idx, geo.k_pool, geo.k_val, geo.n_val).mean(axis=1)
    return CohortValues(r_train, h_train, r_val, h_val, omega)
