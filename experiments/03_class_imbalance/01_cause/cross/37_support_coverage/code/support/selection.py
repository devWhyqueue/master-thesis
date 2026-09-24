"""The three S-arm row picks (PLAN.md "Arms per (split, draw)"), all choosing the same per-patient
quota from the same frozen B pool: ``random`` is S100's own nested-prefix rows (prefix_indices,
called directly from ``support.shard``), ``coverage`` is greedy quota facility location, and
``redundant`` is the per-patient nearest-to-centre pick.
"""

from __future__ import annotations

import numpy as np

from support.coverage import normalize

__all__ = ["prefix_indices", "greedy_quota_coverage", "redundant_pick"]


def prefix_indices(patient_idx: np.ndarray, quota: list[int]) -> np.ndarray:
    """The first ``quota[p]`` pool rows of each patient (S100's own nested round-robin prefix)."""
    out: list[int] = []
    for p, q in enumerate(quota):
        out.extend(np.flatnonzero(patient_idx == p)[:q].tolist())
    return np.asarray(out, dtype=np.int64)


def _eligible(
    remaining: list[int], patient_idx: np.ndarray, taken: np.ndarray
) -> np.ndarray:
    """Rows not yet taken whose patient still has quota left."""
    has_quota = np.array([remaining[p] > 0 for p in patient_idx])
    return has_quota & ~taken


def greedy_quota_coverage(
    x: np.ndarray, patient_idx: np.ndarray, quota: list[int]
) -> np.ndarray:
    """Greedy farthest-point (max-min) row pick over the whole pool, under per-patient quotas.

    Seeds from the row farthest (cosine) from the pool centroid, then repeatedly adds the row with
    the largest distance to the nearest already-selected row, among rows whose patient still has
    quota left. This maximizes coverage of the pool, not of any one patient's own rows.
    """
    e = normalize(x)
    remaining = list(quota)
    taken = np.zeros(len(e), dtype=bool)
    min_dist = np.full(len(e), np.inf)
    seed_scores = 1.0 - e @ normalize(e.mean(axis=0))
    selected: list[int] = []
    for _ in range(int(sum(quota))):
        eligible = _eligible(remaining, patient_idx, taken)
        if not eligible.any():
            raise RuntimeError("Quota exceeds pool availability")
        scores = seed_scores if not selected else min_dist
        candidate = int(np.argmax(np.where(eligible, scores, -np.inf)))
        selected.append(candidate)
        taken[candidate] = True
        remaining[patient_idx[candidate]] -= 1
        min_dist = np.minimum(min_dist, 1.0 - e @ e[candidate])
    return np.asarray(selected, dtype=np.int64)


def redundant_pick(
    x: np.ndarray, patient_idx: np.ndarray, quota: list[int], centre: np.ndarray
) -> np.ndarray:
    """Per patient, the ``quota[p]`` rows nearest (cosine) to the patient-balanced pool centre."""
    e = normalize(x)
    c = normalize(centre)
    dist = 1.0 - e @ c
    selected: list[int] = []
    for p, q in enumerate(quota):
        rows = np.flatnonzero(patient_idx == p)
        order = rows[np.argsort(dist[rows], kind="stable")]
        selected.extend(order[:q].tolist())
    return np.asarray(selected, dtype=np.int64)
