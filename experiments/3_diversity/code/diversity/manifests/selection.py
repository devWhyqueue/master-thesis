"""Narrow/wide selection primitives: nearest-to-mean and greedy farthest-point sampling."""

from __future__ import annotations

import numpy as np

__all__ = ["select_narrow", "select_wide"]


def _select_narrow(
    features: np.ndarray, feature_index: np.ndarray, n: int
) -> np.ndarray:
    """Positional indices of the ``n`` rows nearest the slot mean; ties by ``feature_index``."""
    mean = features.mean(axis=0)
    distance = np.linalg.norm(features - mean, axis=1)
    order = np.lexsort((feature_index, distance))
    return order[:n]


def select_narrow(
    features: np.ndarray, feature_index: np.ndarray, n: int
) -> np.ndarray:
    """Public wrapper over :func:`_select_narrow` for tests and ``check.py``."""
    return _select_narrow(features, feature_index, n)


def _select_wide(features: np.ndarray, feature_index: np.ndarray, n: int) -> np.ndarray:
    """Greedy farthest-point sampling seeded at the row nearest the mean.

    Ties (the seed and each subsequent pick) break by smallest
    ``feature_index``, matching :func:`_select_narrow`'s tie rule.
    """
    m = len(features)
    if n >= m:
        return np.arange(m)
    mean = features.mean(axis=0)
    dist_to_mean = np.linalg.norm(features - mean, axis=1)
    seed = int(np.lexsort((feature_index, dist_to_mean))[0])
    selected = [seed]
    min_dist = np.linalg.norm(features - features[seed], axis=1)
    min_dist[seed] = -np.inf
    for _ in range(n - 1):
        next_idx = int(np.lexsort((feature_index, -min_dist))[0])
        selected.append(next_idx)
        min_dist = np.minimum(
            min_dist, np.linalg.norm(features - features[next_idx], axis=1)
        )
        min_dist[next_idx] = -np.inf
    return np.asarray(selected)


def select_wide(features: np.ndarray, feature_index: np.ndarray, n: int) -> np.ndarray:
    """Public wrapper over :func:`_select_wide` for tests and ``check.py``."""
    return _select_wide(features, feature_index, n)
