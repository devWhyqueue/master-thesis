"""The directed pair-separation score and its argmax/argmin local search.

For classes ``c != d`` at ranks ``r_c``, ``r_d`` under assignment ``pi``::

    S(pi) = sum_{c != d}  w[c, d] * h[c] * max(z[r_d] - z[r_c], 0)

``w[c, d]`` is the row-normalized share of c's off-diagonal r1 test predictions landing on d;
``h[c]`` is c's r1 recall (headroom); ``z[r]`` is the log-count deviation at rank r. Reversing an
order preserves the *undirected* rank-gap sum but changes ``S``: the score rewards separating
confusion partners **and** putting the higher-headroom member underneath.

``search`` reports *a* worst/mildest assignment found by pairwise-swap hill climbing with random
restarts, not a proven optimum over the full 30! permutation space (as the exp-31 plan requires).
"""

from __future__ import annotations

import numpy as np

__all__ = ["confusion_weights", "score", "search"]


def confusion_weights(confusion: np.ndarray) -> np.ndarray:
    """Row-normalized off-diagonal confusion share: ``w[c, d]`` = c's off-diagonal mass on d."""
    off = confusion.astype(np.float64).copy()
    np.fill_diagonal(off, 0.0)
    row_sum = off.sum(axis=1, keepdims=True)
    return np.divide(off, row_sum, out=np.zeros_like(off), where=row_sum > 0)


def _z_of_perm(perm: np.ndarray, z: np.ndarray) -> np.ndarray:
    """(C,) each class's assigned z, from its rank under ``perm`` (``perm[rank] = class index``)."""
    rank_of = np.empty_like(perm)
    rank_of[perm] = np.arange(len(perm))
    return z[rank_of]


def score(perm: np.ndarray, w: np.ndarray, h: np.ndarray, z: np.ndarray) -> float:
    """S(pi) for one full class-to-rank assignment ``perm`` (``perm[rank] = class index``)."""
    zc = _z_of_perm(perm, z)
    gain = np.clip(zc[None, :] - zc[:, None], 0.0, None)
    return float(np.sum(w * h[:, None] * gain))


def _hill_climb(
    perm: np.ndarray, w: np.ndarray, h: np.ndarray, z: np.ndarray, maximize: bool
) -> tuple[np.ndarray, float]:
    """Steepest pairwise-swap ascent/descent from one starting permutation to a local optimum."""
    perm = perm.copy()
    best = score(perm, w, h, z)
    k = len(perm)
    improved = True
    while improved:
        improved = False
        best_pair = None
        for i in range(k):
            for j in range(i + 1, k):
                perm[i], perm[j] = perm[j], perm[i]
                candidate = score(perm, w, h, z)
                perm[i], perm[j] = perm[j], perm[i]
                if (candidate > best) if maximize else (candidate < best):
                    best, best_pair = candidate, (i, j)
                    improved = True
        if best_pair is not None:
            i, j = best_pair
            perm[i], perm[j] = perm[j], perm[i]
    return perm, best


def search(
    w: np.ndarray,
    h: np.ndarray,
    z: np.ndarray,
    maximize: bool,
    seed: int,
    n_restarts: int,
) -> np.ndarray:
    """Best permutation over ``n_restarts`` random-start pairwise-swap hill climbs."""
    k = len(h)
    rng = np.random.default_rng(seed)
    best_perm, best_score = None, None
    for _ in range(n_restarts):
        perm, s = _hill_climb(rng.permutation(k), w, h, z, maximize)
        if best_score is None or ((s > best_score) if maximize else (s < best_score)):
            best_perm, best_score = perm, s
    assert best_perm is not None
    return best_perm
