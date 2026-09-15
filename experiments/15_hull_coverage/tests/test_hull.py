"""Unit tests for hull geometry, design rotation, cohort search, and the manipulation check."""

from __future__ import annotations

import numpy as np
import pytest

from similarity.geometry import ClassGeometry, omega_of

from hull.checks import manipulation_check
from hull.design import CELLS, Target, offsets
from hull.geometry import HullGeometry, cohort_values, hull_residuals
from hull.search import loss, search_cohort


def _toy_geometry(seed: int = 0, p: int = 15, v: int = 6, d: int = 8):
    rng = np.random.default_rng(seed)
    pool, val = rng.normal(size=(p, d)), rng.normal(size=(v, d))
    e_pool = pool / np.linalg.norm(pool, axis=1, keepdims=True)
    e_val = val / np.linalg.norm(val, axis=1, keepdims=True)
    dev = pool - pool.mean(axis=0)
    base = ClassGeometry(
        pool=[f"p{i:02d}" for i in range(p)],
        d_pool=1.0 - e_pool @ e_pool.T,
        d_val=1.0 - e_val @ e_pool.T,
        gram=dev @ dev.T,
        tau2=float(dev.var(axis=0).sum()),
        rho=0.1,
    )
    geo = HullGeometry(base, pool @ pool.T, val @ pool.T, (val**2).sum(axis=1))
    return geo, pool, val


def _explicit_hull(pool: np.ndarray, cohort: np.ndarray, ref: np.ndarray) -> np.ndarray:
    members = pool[cohort]
    centre = members.mean(axis=0)
    basis = (members - centre).T
    z = ref - centre
    coef, *_ = np.linalg.lstsq(basis, z.T, rcond=None)
    resid = z - (basis @ coef).T
    return (resid**2).sum(axis=1) / (z**2).sum(axis=1)


def test_hull_residuals_match_explicit_projection():
    """Gram-based hull residuals equal an explicit least-squares projection."""
    geo, pool, val = _toy_geometry()
    idx = np.array([[0, 3, 5, 7, 9], [1, 2, 4, 6, 8]])
    got = hull_residuals(idx, geo.k_pool, geo.k_val, geo.n_val)
    expected = np.stack([_explicit_hull(pool, row, val) for row in idx])
    assert got == pytest.approx(expected, abs=1e-8)


def test_cohort_values_leave_members_out_of_pool_reference():
    """Pool coverage and hull means exclude cohort members; omega matches exp-14's formula."""
    geo, pool, _ = _toy_geometry()
    cohort = np.array([2, 4, 11, 13, 14])
    values = cohort_values(geo, cohort[None, :])
    others = np.setdiff1d(np.arange(len(pool)), cohort)
    r_expected = geo.base.d_pool[np.ix_(others, cohort)].min(axis=1).mean()
    h_expected = _explicit_hull(pool, cohort, pool[others]).mean()
    assert values.r_train[0] == pytest.approx(r_expected)
    assert values.h_train[0] == pytest.approx(h_expected, abs=1e-8)
    assert values.omega[0] == pytest.approx(omega_of(geo.base, cohort))


@pytest.mark.parametrize("g", [5, 10])
def test_offsets_balance_cells_and_rotate_through_all(g: int):
    """Cells occur about equally often, and each class visits every cell in one rotation."""
    n_cells = len(CELLS[g])
    offs = offsets(0, g, 30)
    counts = np.bincount(offs, minlength=n_cells)
    assert counts.max() - counts.min() <= 1
    for o in offs:
        assert {(int(o) + d) % n_cells for d in range(35)} == set(range(n_cells))


def test_search_cohort_not_worse_than_best_candidate():
    """Swap search never ends above the best starting candidate's loss."""
    geo, _, _ = _toy_geometry(seed=1)
    rng = np.random.default_rng(2)
    cands = [rng.choice(15, size=5, replace=False) for _ in range(20)]
    target = Target(r=0.9, h=0.8, omega=0.0, tol_r=0.005, tol_h=0.01, tol_omega=0.01)
    best_start = loss(geo, np.stack(cands), target).min()
    result = search_cohort(geo, cands, target)
    assert loss(geo, result[None, :], target)[0] <= best_start + 1e-12


def _check_rows(hull_spread: float, seed: int = 0) -> list[dict]:
    """Designed rows at both patient counts, achieved values near target plus noise."""
    rng = np.random.default_rng(seed)
    rows = []
    for g in (5, 10):
        for cell in (c for c in CELLS[g] if c.mean_level is not None):
            for c_name in ("a", "b", "c"):
                for _ in range(4):
                    r_t, h_t = 0.3 + 0.05 * cell.mean_level, 0.8 + 0.15 * cell.hull_level
                    r, h = r_t + rng.normal(0, 0.002), 0.8 + hull_spread * 0.15 * cell.hull_level + rng.normal(0, 0.004)
                    rows.append({
                        "class": c_name, "g": g, "mean_level": cell.mean_level,
                        "hull_level": cell.hull_level, "r_target": r_t, "h_target": h_t,
                        "omega_target": 0.0, "delta_r": 0.05, "delta_h": 0.15,
                        "r_train": r, "h_train": h, "r_val": r, "h_val": h,
                        "omega": rng.normal(0, 0.005),
                    })
    return rows


def test_manipulation_check_passes_on_target_and_fails_without_hull_spread():
    """The check passes for on-target cohorts and fails when the hull levels collapse."""
    assert manipulation_check(_check_rows(hull_spread=1.0))["pass"]
    assert not manipulation_check(_check_rows(hull_spread=0.0))["pass"]
