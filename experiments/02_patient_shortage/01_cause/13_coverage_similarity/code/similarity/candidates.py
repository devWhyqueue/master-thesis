"""Candidate cohort generation: random, dispersed-prefix, and farthest-first (report App. A)."""

from __future__ import annotations

import numpy as np

from composition.allocation import dispersed_patients

from similarity import N_ANCHORS, N_RANDOM
from similarity.geometry import ClassGeometry

__all__ = ["candidates"]


def _random_candidates(
    rng: np.random.Generator, pool_size: int, g: int, n: int
) -> list[np.ndarray]:
    return [rng.choice(pool_size, size=g, replace=False) for _ in range(n)]


def _farthest_first(anchor: int, geo: ClassGeometry, g: int) -> np.ndarray:
    """Greedy farthest-first cohort: repeatedly add the patient farthest from the set."""
    pool_ids = np.asarray(geo.pool)
    selected = [anchor]
    min_dist = geo.d_pool[:, anchor].copy()
    for _ in range(g - 1):
        mask = np.ones(len(pool_ids), dtype=bool)
        mask[selected] = False
        candidates_left = np.flatnonzero(mask)
        order = np.lexsort((pool_ids[candidates_left], -min_dist[candidates_left]))
        nxt = int(candidates_left[order[0]])
        selected.append(nxt)
        min_dist = np.minimum(min_dist, geo.d_pool[:, nxt])
    return np.array(selected)


def _dispersed_prefix(anchor: int, geo: ClassGeometry, g: int) -> np.ndarray:
    """The g-prefix of facility-location selection from one anchor (prefix-consistent)."""
    chosen = dispersed_patients(geo.pool[anchor], geo.pool, geo.d_pool)[:g]
    position = {p: i for i, p in enumerate(geo.pool)}
    return np.array([position[p] for p in chosen])


def candidates(geo: ClassGeometry, g: int, seed: int) -> list[np.ndarray]:
    """Random, dispersed-prefix, and farthest-first candidate cohorts of size g."""
    rng = np.random.default_rng(seed)
    pool_size = len(geo.pool)
    out = _random_candidates(rng, pool_size, g, N_RANDOM)
    anchors = rng.choice(pool_size, size=min(N_ANCHORS, pool_size), replace=False)
    for anchor in anchors:
        out.append(_dispersed_prefix(int(anchor), geo, g))
        out.append(_farthest_first(int(anchor), geo, g))
    return out
