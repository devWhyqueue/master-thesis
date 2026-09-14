"""The twenty-cell design grid: cell assignment and coverage/similarity targets."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from similarity import N_RANDOM
from similarity.geometry import ClassGeometry, omega_of, r_of

__all__ = [
    "Cell",
    "CELLS",
    "N_CELLS",
    "SEARCH_BASE_SEED",
    "Targets",
    "offsets",
    "targets",
]

# Fresh seed: exp-13 and its checks used 20260914 with +0/+1/+2 offsets.
SEARCH_BASE_SEED: int = 20260915

_OMEGA_STEP: float = 0.04


class Cell(NamedTuple):
    """One of the twenty (G, m) x (r_level, omega_level) design cells (report Table "cells")."""

    g: int
    m: int
    r_level: int | None  # None for the two random cells
    omega_level: int | None


def _designed_cells(g: int, m: int) -> tuple[Cell, ...]:
    return tuple(Cell(g, m, *divmod(k, 3)) for k in range(9))


CELLS: tuple[Cell, ...] = (
    *_designed_cells(5, 32),
    *_designed_cells(10, 16),
    Cell(5, 32, None, None),
    Cell(10, 16, None, None),
)
N_CELLS: int = len(CELLS)  # 20


class Targets(NamedTuple):
    """Coverage and similarity targets shared by both patient counts (report Table "cells")."""

    r_levels: tuple[float, float, float]
    omega_levels: tuple[float, float, float]
    delta_prime: float


def _r_bar(geo: ClassGeometry, candidates: list[np.ndarray]) -> float:
    """Mean training-pool coverage distance of the first N_RANDOM random candidates."""
    return float(np.mean([r_of(geo.d_pool, c) for c in candidates[:N_RANDOM]]))


def _omega_median(geo: ClassGeometry, candidates: list[np.ndarray]) -> float:
    """Median between-patient similarity of the first N_RANDOM random candidates."""
    return float(np.median([omega_of(geo, c) for c in candidates[:N_RANDOM]]))


def targets(
    geo: ClassGeometry, cands5: list[np.ndarray], cands10: list[np.ndarray]
) -> Targets:
    """Coverage and similarity targets for one (split, class, draw) (report Sec. "cohorts")."""
    r_bar5 = _r_bar(geo, cands5)
    r_bar10 = _r_bar(geo, cands10)
    delta_prime = r_bar5 - r_bar10
    omega_med10 = _omega_median(geo, cands10)
    return Targets(
        r_levels=(r_bar10, r_bar10 + delta_prime / 2.0, r_bar10 + delta_prime),
        omega_levels=(
            omega_med10 - _OMEGA_STEP,
            omega_med10,
            omega_med10 + _OMEGA_STEP,
        ),
        delta_prime=delta_prime,
    )


def offsets(split_idx: int, n_classes: int = 30) -> np.ndarray:
    """Seeded per-class cell offsets: a shuffle of the 20 cells plus a random subset.

    Every offset in ``[0, N_CELLS)`` occurs once or twice among the
    ``n_classes`` classes, and class ``c`` sees cell ``(offsets(s)[c] + d) %
    N_CELLS`` on draw ``d`` (report Sec. "cohorts").
    """
    if n_classes < N_CELLS:
        raise ValueError(f"n_classes must be >= {N_CELLS}")
    rng = np.random.default_rng(SEARCH_BASE_SEED + 1_000_003 * split_idx)
    full = rng.permutation(N_CELLS)
    extra = rng.choice(N_CELLS, size=n_classes - N_CELLS, replace=False)
    return np.concatenate([full, extra])
