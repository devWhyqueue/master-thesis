"""Coverage deficit c(S) and centre error eps(S) (PLAN.md "Measures"): both cosine distances on
L2-normalized features, scaled by the class pool's own median pairwise distance so a 7-class and a
30-class dataset are comparable.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "normalize",
    "median_pairwise_distance",
    "coverage_deficit",
    "centre_error",
    "patient_balanced_mean",
]


def normalize(x: np.ndarray) -> np.ndarray:
    """L2-normalize each row; ``x`` may be 1-D (one vector) or 2-D (a stack of rows)."""
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    if np.any(norm == 0.0):
        raise ValueError("Zero-norm row; cannot normalize")
    return x / norm


def median_pairwise_distance(pool_x: np.ndarray) -> float:
    """This pool's own median pairwise cosine distance, the scale PLAN.md's measures divide by."""
    e = normalize(pool_x)
    d = 1.0 - e @ e.T
    iu = np.triu_indices(len(e), k=1)
    return float(np.median(d[iu]))


def coverage_deficit(pool_x: np.ndarray, selected_idx: np.ndarray) -> float:
    """c(S) = Q0.90 over the pool of min cosine distance to the selection, over the pool's median."""
    e = normalize(pool_x)
    sel = e[selected_idx]
    nearest = (1.0 - e @ sel.T).min(axis=1)
    return float(np.quantile(nearest, 0.90)) / median_pairwise_distance(pool_x)


def patient_balanced_mean(pool_x: np.ndarray, patient_idx: np.ndarray) -> np.ndarray:
    """Mean-of-patient-means over the whole pool (PLAN.md "patient-balanced class centre")."""
    n_patients = int(patient_idx.max()) + 1
    means = [pool_x[patient_idx == p].mean(axis=0) for p in range(n_patients)]
    return np.mean(means, axis=0)


def centre_error(
    pool_x: np.ndarray, selected_idx: np.ndarray, centre: np.ndarray
) -> float:
    """eps(S) = cosine distance of the selection's mean from ``centre``, over the pool's median."""
    e = normalize(pool_x)
    sel_mean = normalize(e[selected_idx].mean(axis=0))
    c = normalize(centre)
    dist = 1.0 - float(sel_mean @ c)
    return dist / median_pairwise_distance(pool_x)
