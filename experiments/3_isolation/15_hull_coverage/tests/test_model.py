"""Unit tests for the block fixed-effects estimator, its parts, and the precision simulation."""

from __future__ import annotations

import numpy as np
import pytest

from hull.inference import N_STUDY_REPLICATES
from hull.inference.model import CellMeans, estimate, parts, prediction
from hull.inference.simulate import CheckedSplit, Slopes, Templates, study_halfwidths


def _dummy_regression(y, x, block_split, w):
    rows_x, rows_y, rows_w, sc_ids, b_ids = [], [], [], [], []
    for b in range(x.shape[0]):
        for f in range(2):
            for c in range(x.shape[2]):
                rows_x.append(x[b, f, c])
                rows_y.append(y[b, f, c, 0])
                rows_w.append(w[b, 0])
                sc_ids.append((int(block_split[b]), c))
                b_ids.append(b)
    dummies = [
        np.array([[1.0 if i == u else 0.0 for u in sorted(set(ids))] for i in ids])
        for ids in (sc_ids, b_ids)
    ]
    design = np.concatenate([np.array(rows_x), *dummies], axis=1)
    sqrt_w = np.sqrt(np.array(rows_w))
    coef, *_ = np.linalg.lstsq(design * sqrt_w[:, None], np.array(rows_y) * sqrt_w, rcond=None)
    return coef[: x.shape[-1]]


def test_estimate_matches_dummy_variable_regression():
    """Closed-form block/split-class demeaning equals dummy-variable weighted least squares."""
    rng = np.random.default_rng(0)
    block_split = np.repeat(np.arange(2), 3)
    x = rng.normal(size=(6, 2, 4, 4))
    x[:, :, :, 3] = np.array([0.0, 1.0])[None, :, None]
    y = rng.normal(size=(6, 2, 4, 1))
    w = rng.integers(1, 5, size=(6, 1)).astype(float)
    beta = estimate(y, x, block_split, w)
    assert beta[0] == pytest.approx(_dummy_regression(y, x, block_split, w), abs=1e-6)


def test_parts_and_prediction_use_random_cell_differences():
    """Parts multiply slopes by ten-minus-five differences; prediction omits the patient-count term."""
    beta = np.array([[-40.0, -20.0, -30.0, 2.0]])
    means = {5: CellMeans(0.40, 0.90, 0.0), 10: CellMeans(0.35, 0.80, -0.01)}
    pts = parts(beta, means)
    assert pts["C"][0] == pytest.approx(2.0)
    assert pts["H"][0] == pytest.approx(2.0)
    assert pts["S"][0] == pytest.approx(0.3)
    assert pts["P"][0] == pytest.approx(2.0)
    assert prediction(beta, means[10], CellMeans(0.30, 0.70, -0.01))[0] == pytest.approx(4.0)


def test_simulated_study_without_noise_has_zero_halfwidth():
    """Noise-free templates make every replicate identical, so all interval half-widths vanish."""
    rng = np.random.default_rng(3)
    n_classes, n_checked = 3, 5
    checked = []
    for _ in range(3):
        x = rng.normal(size=(n_checked, 2, n_classes, 3))
        random = np.zeros((n_checked, 2, n_classes), dtype=bool)
        random[:, :, 0] = True
        checked.append(CheckedSplit(x, random))
    flat = np.full((4, n_classes, N_STUDY_REPLICATES), 60.0)
    templates = Templates({5: np.full(n_classes, 60.0), 10: np.full(n_classes, 60.0)}, {5: [flat] * 3, 10: [flat] * 3})
    hw = study_halfwidths(rng, checked, templates, 5, Slopes(-40.0, -20.0, 0.0, 3.0), kappa=1.0)
    assert set(hw) == {"C", "H", "S", "P"}
    assert all(v == pytest.approx(0.0, abs=1e-6) for v in hw.values())
