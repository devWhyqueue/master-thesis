"""Class-recall model with split-class and block (split x draw) fixed effects, and its parts.

Every block holds one G=5 and one G=10 fit over the same classes, so each
(block, class) cell has exactly two rows and the two-way demeaning is exact
in closed form. A per-fit intercept would absorb the patient-count indicator,
which is constant within a fit; the block intercept does not.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from decomposition.model import draw_weights, reading

__all__ = [
    "CellMeans",
    "estimate",
    "random_means",
    "parts",
    "prediction",
    "draw_weights",
    "reading",
]


class CellMeans(NamedTuple):
    """Mean validation coverage distance, hull residual, and similarity of a set of cohorts."""

    r: float
    h: float
    omega: float


def _demean_split(
    x: np.ndarray, y: np.ndarray, w: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """``x`` (B, 2, C, K), ``y`` (B, 2, C, R), ``w`` (B, R) -> demeaned (B, 2, C, R, K) and (B, 2, C, R)."""
    w_sum = w.sum(axis=0)
    b_bar_x, b_bar_y = x.mean(axis=(1, 2)), y.mean(axis=(1, 2))
    c_bar_x = np.einsum("br,bfck->crk", w, x) / (2.0 * w_sum)[None, :, None]
    c_bar_y = np.einsum("br,bfcr->cr", w, y) / (2.0 * w_sum)[None, :]
    g_bar_x = np.einsum("br,bk->rk", w, b_bar_x) / w_sum[:, None]
    g_bar_y = np.einsum("br,br->r", w, b_bar_y) / w_sum
    x_t = (
        x[:, :, :, None, :]
        - b_bar_x[:, None, None, None, :]
        - c_bar_x[None, None, :, :, :]
        + g_bar_x[None, None, None, :, :]
    )
    y_t = y - b_bar_y[:, None, None, :] - c_bar_y[None, None] + g_bar_y
    return x_t, y_t


def estimate(
    y: np.ndarray, x: np.ndarray, block_split: np.ndarray, w: np.ndarray
) -> np.ndarray:
    """Weighted two-way FE least squares per replicate: (R, K) coefficients.

    ``y`` is (B, 2, C, R) percent recall, ``x`` is (B, 2, C, K) predictors,
    ``block_split`` is (B,) the split of each block, ``w`` is (B, R) weights.
    """
    n_replicates, n_pred = y.shape[-1], x.shape[-1]
    a_total = np.zeros((n_replicates, n_pred, n_pred))
    b_total = np.zeros((n_replicates, n_pred))
    for s in np.unique(block_split):
        mask = block_split == s
        x_t, y_t = _demean_split(x[mask], y[mask], w[mask])
        a_total += np.einsum("br,bfcri,bfcrj->rij", w[mask], x_t, x_t)
        b_total += np.einsum("br,bfcri,bfcr->ri", w[mask], x_t, y_t)
    return np.linalg.solve(a_total, b_total[..., None])[..., 0]


def random_means(x: np.ndarray, random: np.ndarray) -> dict[int, CellMeans]:
    """Random-cell means per patient count from (B, 2, C, K) predictors and a (B, 2, C) mask."""
    out: dict[int, CellMeans] = {}
    for gi, g in enumerate((5, 10)):
        mask = random[:, gi]
        values = x[:, gi][mask]
        out[g] = CellMeans(*(float(values[:, k].mean()) for k in range(3)))
    return out


def parts(beta: np.ndarray, means: dict[int, CellMeans]) -> dict[str, np.ndarray]:
    """Coverage, hull, similarity, and patient-count parts of the random 5-to-10 gap; positive favours ten."""
    lo, hi = means[5], means[10]
    return {
        "C": beta[:, 0] * (hi.r - lo.r),
        "H": beta[:, 1] * (hi.h - lo.h),
        "S": beta[:, 2] * (hi.omega - lo.omega),
        "P": beta[:, 3],
    }


def prediction(beta: np.ndarray, ten: CellMeans, twenty: CellMeans) -> np.ndarray:
    """Predicted 10-to-20 gap from coverage, hull, and similarity alone (no patient-count term)."""
    return (
        beta[:, 0] * (twenty.r - ten.r)
        + beta[:, 1] * (twenty.h - ten.h)
        + beta[:, 2] * (twenty.omega - ten.omega)
    )
