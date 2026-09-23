"""Pooled WLS: basis(z) tensor-producted with dataset/covariate indicators, batched over
bootstrap replicates (the same pinv pattern as ``permutation.model.fit_piecewise``, generalized
from a per-class fit to one regression pooling every class and both datasets).
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np

from breadth.analyze.secondary import pack_estimate

from classprops import (
    CLASS_COMPOSITION_G1_FACTOR,
    CLASS_COMPOSITION_SHRINK_MIN,
    DATASET_SPECIFIC_G1_FACTOR,
    DATASET_SPECIFIC_SHRINK_MAX,
)
from classprops.pool import Observations

__all__ = [
    "DatasetCovariates",
    "FitResult",
    "gather_covariates",
    "design",
    "wls_batched",
    "predict",
    "fit_model",
    "fit_pool",
    "bracs_effect",
    "shrink_share",
    "label_outcome",
]


class DatasetCovariates(NamedTuple):
    """One dataset's cross-fitted (headroom, margin), gathered per observation row."""

    h: np.ndarray  # (N, R)
    m: np.ndarray  # (N, R)


def gather_covariates(
    obs: Observations, h_arr: np.ndarray, m_arr: np.ndarray
) -> DatasetCovariates:
    """Index the (S, C, R) covariate arrays by each observation's own (split, class)."""
    return DatasetCovariates(
        h=h_arr[obs.split_idx, obs.class_idx, :],
        m=m_arr[obs.split_idx, obs.class_idx, :],
    )


def _basis(z_nr: np.ndarray) -> np.ndarray:
    """(N, R, 3): [1, z+, z-]."""
    return np.stack(
        [np.ones_like(z_nr), np.clip(z_nr, 0, None), np.clip(z_nr, None, 0)], axis=-1
    )


def design(
    z: np.ndarray,
    h: np.ndarray,
    m: np.ndarray,
    b: float,
    n_replicates: int,
    covariates: tuple[str, ...],
) -> np.ndarray:
    """(N, R, P) design: basis(z) tensor-producted with [1, B] and, if requested, h and m.

    ``covariates`` selects which of ``("h", "m")`` also modulate the basis; ``()`` gives M0's
    ``basis x (1 + B)`` (6 columns), ``("h", "m")`` gives M1's ``basis x (1 + B + h + m)`` (12).
    """
    n = z.shape[0]
    basis = _basis(np.broadcast_to(z[:, None], (n, n_replicates)))  # (N, R, 3)
    b_nr = np.full((n, n_replicates, 1), float(b))
    cols = [basis, basis * b_nr]
    if "h" in covariates:
        cols.append(basis * h[..., None])
    if "m" in covariates:
        cols.append(basis * m[..., None])
    return np.concatenate(cols, axis=-1)


def wls_batched(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Per-replicate weighted least squares via Moore-Penrose pinv. x:(N,R,P), y,w:(N,R) -> (R,P)."""
    a = np.einsum("nr,nrp,nrq->rpq", w, x, x)
    b = np.einsum("nr,nrp,nr->rp", w, x, y)
    return np.einsum("rpq,rq->rp", np.linalg.pinv(a), b)


def predict(beta: np.ndarray, x: np.ndarray) -> np.ndarray:
    """(N, R): design x times per-replicate coefficients."""
    return np.einsum("rp,nrp->nr", beta, x)


class FitResult(NamedTuple):
    """One model's fitted coefficients and the BRACS-side dataset effect G."""

    beta: np.ndarray  # (R, P)
    g: np.ndarray  # (R,), BA pp


def fit_model(
    tcga_obs: Observations,
    tcga_cov: DatasetCovariates,
    bracs_obs: Observations,
    bracs_cov: DatasetCovariates,
    covariates: tuple[str, ...],
) -> FitResult:
    """Pool both datasets' observations, fit the WLS, and compute BRACS's own dataset effect."""
    n_replicates = tcga_obs.delta.shape[1]
    x_t = design(tcga_obs.z, tcga_cov.h, tcga_cov.m, 0.0, n_replicates, covariates)
    x_b = design(bracs_obs.z, bracs_cov.h, bracs_cov.m, 1.0, n_replicates, covariates)
    x = np.concatenate([x_t, x_b], axis=0)
    y = np.concatenate([tcga_obs.delta, bracs_obs.delta], axis=0)
    w = np.concatenate([tcga_obs.draw_weight, bracs_obs.draw_weight], axis=0)
    beta = wls_batched(x, y, w)
    g = bracs_effect(beta, bracs_obs, bracs_cov, covariates)
    return FitResult(beta, g)


def fit_pool(
    tcga_obs: Observations,
    tcga_cov: DatasetCovariates,
    bracs_obs: Observations,
    bracs_cov: DatasetCovariates,
) -> dict[str, Any]:
    """M0/M1, the BRACS dataset residual G, and the shrink share, for one (pool, rho)."""
    m0 = fit_model(tcga_obs, tcga_cov, bracs_obs, bracs_cov, covariates=())
    m1 = fit_model(tcga_obs, tcga_cov, bracs_obs, bracs_cov, covariates=("h", "m"))
    shrink = shrink_share(m0.g, m1.g)
    return {
        "G0": pack_estimate(m0.g),
        "G1": pack_estimate(m1.g),
        "shrink_share": pack_estimate(shrink),
    }


def bracs_effect(
    beta: np.ndarray,
    bracs_obs: Observations,
    bracs_cov: DatasetCovariates,
    covariates: tuple[str, ...],
) -> np.ndarray:
    """G_k (BA pp): weighted mean over BRACS observations of -[yhat(B=1) - yhat(B=0)]."""
    n_replicates = beta.shape[0]
    x1 = design(bracs_obs.z, bracs_cov.h, bracs_cov.m, 1.0, n_replicates, covariates)
    x0 = design(bracs_obs.z, bracs_cov.h, bracs_cov.m, 0.0, n_replicates, covariates)
    delta_pred = predict(beta, x1) - predict(beta, x0)  # (N, R)
    w = bracs_obs.draw_weight
    return -(w * delta_pred).sum(axis=0) / w.sum(axis=0)


def shrink_share(g0: np.ndarray, g1: np.ndarray) -> np.ndarray:
    """(R,): 1 - G1 / G0, per replicate."""
    return 1.0 - g1 / g0


def label_outcome(
    shrink: dict[str, float], g0_point: float, g1_estimate: dict[str, float]
) -> str:
    """The pre-registered outcome label from shrink share and G1's interval vs. 0.5*G0."""
    if (
        shrink["point"] >= CLASS_COMPOSITION_SHRINK_MIN
        and g1_estimate["ci_97_5"] < CLASS_COMPOSITION_G1_FACTOR * g0_point
    ):
        return "class_composition"
    if (
        shrink["point"] <= DATASET_SPECIFIC_SHRINK_MAX
        or g1_estimate["ci_2_5"] > DATASET_SPECIFIC_G1_FACTOR * g0_point
    ):
        return "dataset_specific"
    return "partial"
