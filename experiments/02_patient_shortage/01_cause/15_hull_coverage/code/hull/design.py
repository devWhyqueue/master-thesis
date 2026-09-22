"""Design cells per patient count, per (split, class, draw) targets, and class rotation offsets."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from similarity import N_RANDOM

from hull import OMEGA_TOL, R_TOL, SEARCH_BASE_SEED
from hull.geometry import HullGeometry, pool_values

__all__ = ["Cell", "CELLS", "Targets", "Target", "targets", "cell_target", "offsets"]

_MIN_DELTA = 1e-6


class Cell(NamedTuple):
    """One design cell: patient count, patches per patient, and target levels (None if random)."""

    g: int
    m: int
    mean_level: float | None
    hull_level: float | None


def _designed(g: int, m: int, hull_levels: tuple[float, ...]) -> tuple[Cell, ...]:
    return tuple(Cell(g, m, ml, hl) for ml in (0.0, 1.0) for hl in hull_levels)


CELLS: dict[int, tuple[Cell, ...]] = {
    5: (*_designed(5, 32, (0.0, 0.5, 1.0)), Cell(5, 32, None, None)),
    10: (*_designed(10, 16, (0.0, 0.5)), Cell(10, 16, None, None)),
    20: (Cell(20, 8, None, None),),
}


class Targets(NamedTuple):
    """Random-cohort reference values shared by every cell of one (split, class, draw)."""

    r_ran10: float
    delta_r: float
    h_ran10: float
    delta_h: float
    omega_med10: float


class Target(NamedTuple):
    """One cohort's search target and loss scales."""

    r: float
    h: float
    omega: float
    tol_r: float
    tol_h: float
    tol_omega: float


def targets(
    geo: HullGeometry, cands5: list[np.ndarray], cands10: list[np.ndarray]
) -> Targets:
    """Random five- and ten-patient pool values from the first N_RANDOM candidates of each."""
    r5, h5, _ = pool_values(geo, np.stack(cands5[:N_RANDOM]))
    r10, h10, omega10 = pool_values(geo, np.stack(cands10[:N_RANDOM]))
    return Targets(
        r_ran10=float(r10.mean()),
        delta_r=float(r5.mean() - r10.mean()),
        h_ran10=float(h10.mean()),
        delta_h=float(h5.mean() - h10.mean()),
        omega_med10=float(np.median(omega10)),
    )


def cell_target(tg: Targets, cell: Cell) -> Target:
    """A designed cell's (r, h, omega) target; levels are in units of the random 5-vs-10 gaps."""
    if cell.mean_level is None or cell.hull_level is None:
        raise ValueError("Random cells have no search target")
    delta_r = max(abs(tg.delta_r), _MIN_DELTA)
    delta_h = max(abs(tg.delta_h), _MIN_DELTA)
    return Target(
        r=tg.r_ran10 + cell.mean_level * tg.delta_r,
        h=tg.h_ran10 + cell.hull_level * tg.delta_h,
        omega=tg.omega_med10,
        tol_r=R_TOL,
        tol_h=R_TOL * delta_h / delta_r,
        tol_omega=OMEGA_TOL,
    )


def offsets(split_idx: int, g: int, n_classes: int = 30) -> np.ndarray:
    """Seeded per-class cell offsets for patient count ``g``; every cell occurs about equally often.

    Class ``c`` receives cell ``(offsets[c] + d) % len(CELLS[g])`` in draw ``d``.
    """
    n_cells = len(CELLS[g])
    rng = np.random.default_rng(SEARCH_BASE_SEED + 1_000_003 * split_idx + g)
    return rng.permutation(np.resize(np.arange(n_cells), n_classes))
