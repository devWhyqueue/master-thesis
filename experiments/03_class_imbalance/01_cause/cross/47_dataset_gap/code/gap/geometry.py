"""Deterministic patient-balanced sampling and high-dimensional neighbour geometry.

Reference sets come from training patients (only full 32-patch groups, the same patient count in
every class, so every reference class has equal mass); query sets come from the held-out test
patients of the same split and are weighted per patient-class group. Agreement is the share of a
query patch's k nearest reference patches that carry its own class.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd

__all__ = [
    "sample_rows",
    "assert_disjoint",
    "knn_agreement",
    "nearest",
    "chance_adjusted",
    "variance_shares",
]

_QUERY_CHUNK = 2048


def _groups(frame: pd.DataFrame, label: str) -> pd.Series:
    """Rows per patient within one class, in sorted patient order."""
    sub = frame[frame["cancer_type"] == label]
    return cast(pd.Series, sub.groupby("case_id").size().sort_index())


def sample_rows(
    frame: pd.DataFrame,
    n_patients: int,
    n_patches: int,
    seed: int,
    full_groups: bool = False,
) -> list[int]:
    """Up to ``n_patients`` patient-class groups per class, up to ``n_patches`` rows each.

    ``full_groups`` keeps only groups with at least ``n_patches`` rows and takes the same number
    of patients in every class (the smallest feasible count, fixed before any geometry).
    Sampling is without replacement and depends only on ``seed`` and the frame's contents.
    """
    rng = np.random.default_rng(seed)
    labels = sorted(frame["cancer_type"].unique())
    eligible: dict[str, np.ndarray] = {}
    for label in labels:
        sizes = _groups(frame, label)
        keep = (sizes >= n_patches) if full_groups else (sizes > 0)
        eligible[label] = np.asarray(sizes.index)[keep.to_numpy()]
    per_class = min(n_patients, *(len(e) for e in eligible.values()))
    out: list[int] = []
    for label in labels:
        cap = per_class if full_groups else min(n_patients, len(eligible[label]))
        patients = rng.permutation(eligible[label])[:cap]
        sub = frame[frame["cancer_type"] == label]
        for patient in sorted(patients):
            rows = np.asarray(sub.index)[(sub["case_id"] == patient).to_numpy()]
            take = rng.choice(rows, size=min(len(rows), n_patches), replace=False)
            out.extend(int(r) for r in np.sort(take))
    return out


def assert_disjoint(reference: np.ndarray, query: np.ndarray) -> None:
    """Fail if any patient contributes to both the reference and the query set."""
    shared = sorted(set(reference.astype(str)) & set(query.astype(str)))
    if shared:
        raise RuntimeError(f"reference and query share patients: {shared[:5]}")


def _distances(reference: np.ndarray, query: np.ndarray, metric: str) -> np.ndarray:
    """(n_query, n_reference) cosine distance on L2-normalised rows, or raw squared Euclidean."""
    if metric == "cosine":
        r = reference / np.linalg.norm(reference, axis=1, keepdims=True)
        q = query / np.linalg.norm(query, axis=1, keepdims=True)
        return 1.0 - q @ r.T
    if metric == "euclidean":
        return (
            (query**2).sum(axis=1)[:, None]
            + (reference**2).sum(axis=1)[None, :]
            - 2.0 * query @ reference.T
        )
    raise ValueError(f"unknown metric {metric}")


def _neighbour_indices(
    reference: np.ndarray, query: np.ndarray, k: int, metric: str
) -> np.ndarray:
    """(n_query, k) reference indices sorted by distance, computed in query chunks."""
    out = []
    for start in range(0, len(query), _QUERY_CHUNK):
        d = _distances(reference, query[start : start + _QUERY_CHUNK], metric)
        part = np.argpartition(d, k - 1, axis=1)[:, :k] if k < d.shape[1] else None
        part = np.tile(np.arange(d.shape[1]), (len(d), 1)) if part is None else part
        order = np.argsort(np.take_along_axis(d, part, axis=1), axis=1, kind="stable")
        out.append(np.take_along_axis(part, order, axis=1))
    return np.concatenate(out)


def knn_agreement(
    reference: np.ndarray,
    reference_labels: np.ndarray,
    query: np.ndarray,
    query_labels: np.ndarray,
    ks: tuple[int, ...],
    metric: str,
) -> np.ndarray:
    """(n_query, len(ks)) share of each query's k nearest reference patches in its own class."""
    idx = _neighbour_indices(reference, query, max(ks), metric)
    same = reference_labels[idx] == query_labels[:, None]
    running = np.cumsum(same, axis=1) / np.arange(1, idx.shape[1] + 1)
    return running[:, [k - 1 for k in ks]]


def nearest(
    reference: np.ndarray, query: np.ndarray, metric: str
) -> tuple[np.ndarray, np.ndarray]:
    """Index of and distance to the nearest reference row for every query row."""
    d = _distances(reference, query, metric)
    idx = d.argmin(axis=1)
    return idx, d[np.arange(len(d)), idx]


def chance_adjusted(agreement: np.ndarray, n_classes: int) -> np.ndarray:
    """(agreement - 1/K) / (1 - 1/K): 0 at a class-balanced chance level, 1 when perfect."""
    chance = 1.0 / n_classes
    return (agreement - chance) / (1.0 - chance)


def variance_shares(
    x: np.ndarray, patients: np.ndarray, labels: np.ndarray, n_classes: int
) -> np.ndarray:
    """Per class, the share of within-class patch variance that lies within patients."""
    out = np.full(n_classes, np.nan)
    for c in range(n_classes):
        mask = labels == c
        if not mask.any():
            continue
        xc, pc = x[mask], patients[mask].astype(str)
        total = ((xc - xc.mean(axis=0)) ** 2).sum()
        means = pd.DataFrame(xc).groupby(pc).transform("mean").to_numpy()
        out[c] = ((xc - means) ** 2).sum() / total
    return out


def weighted_class_means(
    values: np.ndarray, labels: np.ndarray, weights: np.ndarray, n_classes: int
) -> np.ndarray:
    """Per-class weighted mean of ``values`` (n, m) under unit-mass class weights."""
    return np.stack(
        [np.bincount(labels, weights * v, minlength=n_classes) for v in values.T],
        axis=1,
    )


def summarize_agreement(
    agreement: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    ks: tuple[int, ...],
    n_classes: int,
) -> dict[str, Any]:
    """Patient-weighted per-class and class-mean raw and chance-adjusted agreement per k."""
    per_class = weighted_class_means(agreement, labels, weights, n_classes)
    return {
        f"k{k}": {
            "mean": float(per_class[:, j].mean()),
            "chance_adjusted": float(
                chance_adjusted(per_class[:, j], n_classes).mean()
            ),
            "per_class": per_class[:, j].tolist(),
        }
        for j, k in enumerate(ks)
    }
