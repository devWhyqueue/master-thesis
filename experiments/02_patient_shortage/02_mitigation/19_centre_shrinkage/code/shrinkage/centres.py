"""James-Stein shrinkage of class centres toward the cohort's own grand mean."""

from __future__ import annotations

import numpy as np

from centre.cohort import TrainingTable

__all__ = ["patient_means", "js_weights", "shrunk_centres"]


def patient_means(table: TrainingTable, n_classes: int, g: int) -> np.ndarray:
    """(C, G, d) per-patient mean features, rows ordered class -> patient -> patch."""
    m = len(table.x) // (n_classes * g)
    return table.x.reshape(n_classes, g, m, -1).mean(axis=2)


def js_weights(means: np.ndarray) -> np.ndarray:
    """(C,) positive-part James-Stein weight toward the grand mean.

    ``min(1, ((d - 2) / d) * v_c / ||mu_c - mu_bar||^2)`` with ``v_c = tr(cov_c) / G`` the
    per-class estimation noise from the cohort's own between-patient covariance at dof G - 1.
    """
    g, d = means.shape[1], means.shape[2]
    mu_c = means.mean(axis=1)
    mu_bar = mu_c.mean(axis=0)
    v_c = ((means - mu_c[:, None, :]) ** 2).sum(axis=(1, 2)) / ((g - 1) * g)
    dist_sq = ((mu_c - mu_bar) ** 2).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = np.where(dist_sq > 0.0, ((d - 2) / d) * v_c / dist_sq, np.inf)
    return np.minimum(1.0, raw)


def shrunk_centres(means: np.ndarray, alpha: float) -> np.ndarray:
    """(C, d) centres shrunk toward the grand mean by ``w_c(alpha) = min(1, alpha * js_weights)``."""
    mu_c = means.mean(axis=1)
    mu_bar = mu_c.mean(axis=0)
    w = np.minimum(1.0, alpha * js_weights(means))
    return mu_bar + (1.0 - w)[:, None] * (mu_c - mu_bar)
