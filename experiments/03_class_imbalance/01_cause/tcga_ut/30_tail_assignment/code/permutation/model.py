"""Class-recall model (Part A core): a per-class piecewise-linear own-recall response to
allocation, fit from exp-25's own 30 stored (split, draw) fits, and its predictions across
observed, sampled, and out-of-sample (easy/hard) permutations.

Every stored fit realizes each class's log-count deviation from balance,
``z_c = log2(count_c / (g * BALANCED))``; ``count_c`` is read back from a fit's own stored
``class_counts`` rather than recomputed. Head gain (z > 0) and tail loss (z < 0) are fit
separately -- BRACS's exp-29 own-rank response showed an asymmetric, saturating shape -- so each
class gets three coefficients: an intercept and one slope per side.

For the ratio arms (r10/r100), the per-rank patch count depends only on the shared budget, the
ratio, and the class count (identical, uniform ``available`` cap across classes), never on which
class occupies that rank. So the rank -> count mapping (:func:`ranked_counts`) is one fixed,
permutation-independent vector, and any full permutation's per-class z-vector is just that
vector re-indexed by the permutation. This is what makes the sampled-permutation spread (10,000
samples) and the per-class tail-loss reading (:func:`tail_z`) cheap: no new fits, no I/O.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy.stats import spearmanr

from prevalence import BALANCED, DEPTH
from prevalence.fit import class_counts as _class_counts

__all__ = [
    "Coefficients",
    "ranked_counts",
    "z_of_counts",
    "tail_z",
    "fit_piecewise",
    "predict_delta",
    "predict_d",
    "leave_one_draw_out",
    "sampled_permutation_d",
    "spearman_distribution",
]


def ranked_counts(rho: int, g: int, k: int) -> np.ndarray:
    """(K,) patch count by rank (0 = head/largest) for one ratio arm; independent of assignment."""
    available = [g * DEPTH] * k  # uniform availability cap, same for every class
    identity = np.arange(k)
    return np.asarray(
        _class_counts(f"r{rho}", identity, available, [0] * k, g), dtype=np.float64
    )


def z_of_counts(counts: np.ndarray, g: int) -> np.ndarray:
    """Log2 deviation from the balanced (rho = 1) per-class count."""
    return np.log2(counts / (g * BALANCED))


def tail_z(rho: int, g: int, k: int) -> float:
    """The single z value every class takes when placed at the last (smallest-count) rank."""
    return float(z_of_counts(ranked_counts(rho, g, k), g)[-1])


def _design(z: np.ndarray) -> np.ndarray:
    """(..., 3) predictor row ``[1, max(z, 0), min(z, 0)]``."""
    return np.stack(
        [np.ones_like(z), np.clip(z, 0, None), np.clip(z, None, 0)], axis=-1
    )


class Coefficients(NamedTuple):
    """Per-class piecewise coefficients: intercept, head slope (z > 0), tail slope (z < 0)."""

    a: np.ndarray
    b_pos: np.ndarray
    b_neg: np.ndarray

    def point(self) -> "Coefficients":
        """Replicate-0 (observed point estimate) slice, dropping the replicate axis."""
        return Coefficients(self.a[..., 0], self.b_pos[..., 0], self.b_neg[..., 0])


def fit_piecewise(
    z: np.ndarray, delta: np.ndarray, weight: np.ndarray, robust: bool = False
) -> Coefficients:
    """Per-class weighted LS of own-delta on ``[1, z+, z-]``, batched over bootstrap replicates.

    ``z``/``delta``/``weight`` are (C, N, R): N stacked (arm, fit) observations per class, R
    bootstrap replicate columns (column 0 is the observed point estimate). ``robust`` uses the
    Moore-Penrose pseudo-inverse instead of ``solve`` (never raises on a rank-deficient design,
    e.g. a leave-one-draw-out fold whose remaining rows happen to sit on one side of z = 0).
    """
    x = _design(z)  # (C, N, R, 3)
    a_mat = np.einsum("cnr,cnri,cnrj->crij", weight, x, x)
    b_vec = np.einsum("cnr,cnri,cnr->cri", weight, x, delta)
    if robust:
        beta = np.einsum("crij,crj->cri", np.linalg.pinv(a_mat), b_vec)
    else:
        beta = np.linalg.solve(a_mat, b_vec[..., None])[..., 0]
    return Coefficients(beta[..., 0], beta[..., 1], beta[..., 2])


def predict_delta(coefs: Coefficients, z: np.ndarray) -> np.ndarray:
    """Predicted own-recall delta vs r1. ``z`` broadcasts against the coefficients' shape."""
    if z.ndim < coefs.a.ndim:
        z = z[..., None]
    return (
        coefs.a + coefs.b_pos * np.clip(z, 0, None) + coefs.b_neg * np.clip(z, None, 0)
    )


def predict_d(coefs: Coefficients, z: np.ndarray) -> np.ndarray:
    """Predicted BA damage vs r1: -mean_c predicted own delta."""
    return -predict_delta(coefs, z).mean(axis=0)


def leave_one_draw_out(
    z: np.ndarray,
    delta_point: np.ndarray,
    fit_index: np.ndarray,
    target_col: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Predicted vs. observed r100 D for each held-out (split, draw) fit, refit on the rest.

    ``z``/``delta_point`` are (C, N) point-estimate design/response; ``fit_index`` is (N,), the
    (split, draw) fit each column belongs to; ``target_col`` is (F,), the r100 column index of
    each fit (used both as the held-out prediction target and to read its observed D).
    """
    n_fits = len(target_col)
    predicted = np.full(n_fits, np.nan)
    observed = np.full(n_fits, np.nan)
    z3, delta3 = z[:, :, None], delta_point[:, :, None]
    for fold, col in enumerate(target_col):
        weight = (fit_index != fold).astype(np.float64)
        w3 = np.broadcast_to(weight[None, :, None], z3.shape)
        coefs = fit_piecewise(z3, delta3, w3, robust=True)
        predicted[fold] = predict_d(coefs, z[:, col])[0]
        observed[fold] = -delta_point[:, col].mean()
    return predicted, observed


def sampled_permutation_d(
    coefs_point: Coefficients,
    z_by_rank: np.ndarray,
    rng: np.random.Generator,
    n_samples: int,
) -> np.ndarray:
    """(n_samples,) predicted D of random full permutations, using point-estimate coefficients."""
    k = z_by_rank.shape[0]
    ranks = np.broadcast_to(np.arange(k), (n_samples, k)).copy()
    rng.permuted(ranks, axis=1, out=ranks)
    z_full = z_by_rank[ranks]  # (n_samples, K): rank assigned to each class, per sample
    delta = (
        coefs_point.a[None, :]
        + coefs_point.b_pos[None, :] * np.clip(z_full, 0, None)
        + coefs_point.b_neg[None, :] * np.clip(z_full, None, 0)
    )
    return -delta.mean(axis=1)


def spearman_distribution(x: list[float], y: np.ndarray) -> np.ndarray:
    """Spearman r per replicate column of ``y`` (C, R) against the fixed ``x`` (C,)."""
    return np.array([spearmanr(x, y[:, r])[0] for r in range(y.shape[1])])
