"""Covariance-trace multivariate ICC (Eq. correlation) and cell effective support.

Pure numpy: no dependence on the sampled feature source, so the same code
serves the scalar single-direction measure (features passed as an ``(N, 1)``
projected-score column) and the full-feature measure (``(N, D)``).
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

__all__ = ["ClusterStats", "cluster_stats", "weighted_icc", "cell_effective_support"]


class ClusterStats(NamedTuple):
    """Per-patient sufficient statistics for the covariance-trace ICC estimator."""

    cases: np.ndarray
    n: np.ndarray
    ssw: np.ndarray
    gram: np.ndarray


def cluster_stats(features: np.ndarray, case_ids: np.ndarray) -> ClusterStats:
    """Per-patient size, within-patient SS, and Gram matrix of centered means.

    Patient means are centered on the overall sample mean before the Gram
    matrix is formed, so that ``weighted_icc`` sums small, well-scaled
    quantities rather than differencing two large near-equal numbers.
    """
    cases, inverse, counts = np.unique(
        case_ids, return_inverse=True, return_counts=True
    )
    n = counts.astype(np.float64)
    grand_mean = features.mean(axis=0)
    sums = np.zeros((len(cases), features.shape[1]), dtype=np.float64)
    np.add.at(sums, inverse, features)
    means = sums / n[:, np.newaxis]
    centered_means = means - grand_mean
    gram = centered_means @ centered_means.T
    within_sq = np.sum((features - means[inverse]) ** 2, axis=1)
    ssw = np.zeros(len(cases), dtype=np.float64)
    np.add.at(ssw, inverse, within_sq)
    return ClusterStats(cases=cases, n=n, ssw=ssw, gram=gram)


class _WeightedSums(NamedTuple):
    """Replicate-wise ``(R,)`` reductions of a weighted patient sample."""

    wn: np.ndarray
    h: np.ndarray
    total_n: np.ndarray
    sum_wn2: np.ndarray
    ssw_weighted: np.ndarray


def _weighted_sums(stats: ClusterStats, w: np.ndarray) -> _WeightedSums:
    n = stats.n[np.newaxis, :]
    wn = w * n
    return _WeightedSums(
        wn=wn,
        h=w.sum(axis=1),
        total_n=wn.sum(axis=1),
        sum_wn2=(w * n**2).sum(axis=1),
        ssw_weighted=(w * stats.ssw[np.newaxis, :]).sum(axis=1),
    )


def _require_defined(h: np.ndarray, total_n: np.ndarray) -> None:
    invalid = (h < 2) | (total_n - h <= 0)
    if np.any(invalid):
        raise ValueError(
            f"{int(np.sum(invalid))} replicate(s) have fewer than 2 patients "
            "or no within-patient degrees of freedom"
        )


def _between_sum_of_squares(
    stats: ClusterStats, wn: np.ndarray, total_n: np.ndarray
) -> np.ndarray:
    diag = np.diag(stats.gram)[np.newaxis, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        cross_term = np.where(
            total_n > 0, np.einsum("ri,ij,rj->r", wn, stats.gram, wn) / total_n, 0.0
        )
    return (wn * diag).sum(axis=1) - cross_term


def weighted_icc(stats: ClusterStats, weights: np.ndarray) -> np.ndarray:
    """Raw (unclipped) ICC (Eq. correlation) for each row of integer weights.

    ``weights`` is ``(R, H)`` (or ``(H,)`` for one replicate), aligned with
    ``stats.cases``. A weight of ``k`` counts patient ``i`` as ``k`` separate
    clusters with the same ``n_i``, within-patient SS, and patient mean --
    mathematically identical to duplicating that patient's rows ``k`` times,
    since the between/within sums of squares below are linear or quadratic
    in each patient's contribution and invariant to how patient means were
    centered (the constant cancels in ``ssb``).
    """
    w = np.atleast_2d(np.asarray(weights, dtype=np.float64))
    sums = _weighted_sums(stats, w)
    _require_defined(sums.h, sums.total_n)
    ssb = _between_sum_of_squares(stats, sums.wn, sums.total_n)

    between_ms = ssb / (sums.h - 1.0)
    within_ms = sums.ssw_weighted / (sums.total_n - sums.h)
    m_tilde = (sums.total_n - sums.sum_wn2 / sums.total_n) / (sums.h - 1.0)
    denom = between_ms + (m_tilde - 1.0) * within_ms
    if np.any(denom <= 0):
        raise ValueError("covariance-trace ICC denominator is non-positive")
    return (between_ms - within_ms) / denom


def cell_effective_support(
    class_rho: np.ndarray, cells: list[tuple[int, int]] | tuple[tuple[int, int], ...]
) -> np.ndarray:
    """Class-averaged effective support (Eq. neff) for each cell, over replicates.

    ``class_rho`` is ``(R, C)`` clipped, split-averaged class correlations.
    Returns ``(R, len(cells))``.
    """
    rho = np.asarray(class_rho, dtype=np.float64)
    out = np.empty((rho.shape[0], len(cells)), dtype=np.float64)
    for cell_idx, (g, m) in enumerate(cells):
        design_effect = 1.0 + (float(m) - 1.0) * rho
        out[:, cell_idx] = np.mean((g * m) / design_effect, axis=1)
    return out
