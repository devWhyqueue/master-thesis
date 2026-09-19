"""Directional shrinkage of class centres along the cohort's own between-patient eigenbasis.

Shrinks strongly along eigen-directions where the between-class spread of the cohort's class
means is no larger than the direction's own patient-mean noise, and weakly where it carries
real class separation. Unlike James-Stein shrinkage toward the grand mean (``shrinkage.centres``),
which moves every class by the same scalar fraction of its offset, this moves classes only along
directions the cohort's own patients say are noisy.
"""

from __future__ import annotations

import numpy as np

__all__ = ["directional_weights", "shrunk_centres"]


def directional_weights(
    means: np.ndarray, basis: np.ndarray, eigvals: np.ndarray, g: int
) -> np.ndarray:
    """(k,) empirical-Bayes weight ``n_j / (s_j + n_j)`` per eigen-direction of ``basis``.

    ``n_j = eigvals[j] / g`` is the noise variance of a g-patient class mean along direction j;
    ``s_j = max(0, sum_c z_cj^2 / (C - 1) - n_j)`` is the direction's between-class signal, with
    ``z_cj`` the class centres' deviation from the grand mean projected onto direction j. A zero
    denominator (no signal, no noise) gives weight zero, i.e. no shrinkage along that direction.
    """
    mu_c = means.mean(axis=1)
    mu_bar = mu_c.mean(axis=0)
    z = (mu_c - mu_bar) @ basis.T
    n_j = eigvals / g
    signal = (z**2).sum(axis=0) / (len(mu_c) - 1)
    s_j = np.maximum(0.0, signal - n_j)
    denom = s_j + n_j
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denom > 0.0, n_j / denom, 0.0)


def shrunk_centres(
    means: np.ndarray, basis: np.ndarray, eigvals: np.ndarray, g: int, alpha: float
) -> np.ndarray:
    """(C, d) centres shrunk toward the grand mean along noisy directions of ``basis``.

    ``alpha = 0`` leaves centres unchanged; directions outside ``basis`` (a zero-rank ``basis``
    included) are never touched, since the correction is confined to ``basis``'s row space.
    """
    mu_c = means.mean(axis=1)
    mu_bar = mu_c.mean(axis=0)
    z = (mu_c - mu_bar) @ basis.T
    a = np.minimum(1.0, alpha * directional_weights(means, basis, eigvals, g))
    return mu_c - (z * a) @ basis
