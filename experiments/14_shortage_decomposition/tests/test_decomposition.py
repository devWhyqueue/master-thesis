"""Unit tests for the shortage-decomposition estimator, design grid, and manipulation check."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from decomposition.checks import manipulation_check, spread
from decomposition.design import N_CELLS, offsets
from decomposition.model import estimate, reading


def test_estimate_matches_dummy_variable_regression():
    """The closed-form weighted two-way FE estimate equals dummy-variable lstsq."""
    rng = np.random.default_rng(0)
    n_splits, classes_per_split, fits_per_split = 2, 4, 3
    fit_split = np.repeat(np.arange(n_splits), fits_per_split)
    n_fits, n_classes = len(fit_split), classes_per_split

    x = rng.normal(size=(n_fits, n_classes, 3))
    y = rng.normal(size=(n_fits, n_classes, 1))
    w = rng.integers(1, 5, size=(n_fits, 1)).astype(float)

    beta = estimate(y, x, fit_split, w)

    rows_x, rows_y, rows_w, sc_ids, f_ids = [], [], [], [], []
    for f in range(n_fits):
        for c in range(n_classes):
            rows_x.append(x[f, c])
            rows_y.append(y[f, c, 0])
            rows_w.append(w[f, 0])
            sc_ids.append((int(fit_split[f]), c))
            f_ids.append(f)
    sc_unique, f_unique = sorted(set(sc_ids)), sorted(set(f_ids))
    d_sc = np.array([[1.0 if sc == u else 0.0 for u in sc_unique] for sc in sc_ids])
    d_f = np.array([[1.0 if fi == u else 0.0 for u in f_unique] for fi in f_ids])
    design = np.concatenate([np.array(rows_x), d_sc, d_f], axis=1)
    sqrt_w = np.sqrt(np.array(rows_w))
    coef, *_ = np.linalg.lstsq(design * sqrt_w[:, None], np.array(rows_y) * sqrt_w, rcond=None)

    assert beta[0, :3] == pytest.approx(coef[:3], abs=1e-6)


def test_offsets_each_value_once_or_twice():
    """Every offset in [0, N_CELLS) occurs once or twice among the 30 classes."""
    offs = offsets(0, 30)
    assert len(offs) == 30
    counts = np.bincount(offs, minlength=N_CELLS)
    assert set(counts.tolist()) <= {1, 2}
    assert counts.sum() == 30


def test_offsets_class_visits_all_cells_in_twenty_draws():
    """Every class's offset cycles through all 20 cells exactly once in 20 draws."""
    offs = offsets(1, 30)
    for o in offs:
        cells = {(int(o) + d) % N_CELLS for d in range(N_CELLS)}
        assert cells == set(range(N_CELLS))


def _designed_rows(g: int, r0: float, r1: float, r2: float, o0: float, o1: float, o2: float, delta_prime: float):
    """Nine designed rows for one G on a single class, achieved values equal to targets."""
    r_levels, omega_levels = (r0, r1, r2), (o0, o1, o2)
    rows = []
    for r_level in range(3):
        for omega_level in range(3):
            r_val = r_levels[r_level]
            omega_val = omega_levels[omega_level]
            rows.append(
                {
                    "class": "cls",
                    "g": g,
                    "r_level": r_level,
                    "omega_level": omega_level,
                    "r_target": r_val,
                    "omega_target": omega_val,
                    "delta_prime": delta_prime,
                    "r_train": r_val,
                    "r_val": r_val,
                    "omega": omega_val,
                }
            )
    return rows


def _passing_rows() -> list[dict]:
    rows = []
    for g in (5, 10):
        rows.extend(_designed_rows(g, 0.30, 0.35, 0.40, 0.06, 0.10, 0.14, 0.10))
    return rows


def test_manipulation_check_pass():
    """A well-spread, on-target, orthogonal design passes for both G=5 and G=10."""
    check = manipulation_check(_passing_rows())
    assert check["pass"] is True


def test_manipulation_check_spread_failure():
    """Compressing G=5's r spread below 0.9*Delta' fails the check."""
    rows = _passing_rows()
    for row in rows:
        if row["g"] == 5 and row["r_level"] == 2:
            row["r_train"] = row["r_val"] = 0.31  # spread 0.01 < 0.9*0.10
    spread_result = spread(rows, 5)
    assert spread_result["pass"] is False
    assert manipulation_check(rows)["pass"] is False


def test_manipulation_check_overlap_failure():
    """Shifting every achieved r away from target beyond Delta'/4 fails overlap only."""
    rows = copy.deepcopy(_passing_rows())
    for row in rows:
        row["r_train"] = row["r_val"] = row["r_target"] + 0.05  # miss > Delta'/4 = 0.025
    check = manipulation_check(rows)
    assert check["overlap_pass"] is False
    assert check["pass"] is False
    # The uniform shift cancels in the spread and transfer differences.
    assert spread(rows, 5)["pass"] is True
    assert spread(rows, 10)["pass"] is True


@pytest.mark.parametrize(
    "ci,expected",
    [
        ((1.5, 2.0), "contributes"),
        ((-3.0, -1.5), "counteracts"),
        ((-0.5, 0.9), "at_most_small"),
        ((-0.5, 1.5), "unresolved"),
        ((1.0, 1.0), "unresolved"),  # lower must be strictly > 1
        ((-1.0, -1.0), "at_most_small"),  # upper must be strictly < -1 to counteract
    ],
)
def test_reading_boundaries(ci, expected):
    assert reading(ci) == expected
