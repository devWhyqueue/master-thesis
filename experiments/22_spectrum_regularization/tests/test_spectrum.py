"""Unit tests for spectrum constructions, Bt selection, and record copying."""

from __future__ import annotations

import numpy as np
import pytest

from imbalance_benchmark.common import ensure_dirs, read_run_record, split_paths, write_run_record

from sites import allocation_dir

from centre.arms import whiten
from centre.cohort import TrainingTable
from centre.pool import Pool

import spectrum.fit as fit_mod
from spectrum import ARMS, BETAS
from spectrum.basis import (
    oracle_eigvals,
    oracle_slope,
    spectrum_eigvals,
    within_patient_basis,
)
from spectrum.fit import select_beta


def _pool(basis: np.ndarray, eigvals: np.ndarray) -> Pool:
    """Pool carrying only a between-patient eigenbasis."""
    empty = np.empty(0)
    return Pool([], empty, empty, basis, eigvals, empty, empty)


def test_beta_one_keeps_and_beta_zero_flattens_spectrum():
    """beta = 1 returns raw eigenvalues; beta = 0 whitens every direction by the same scale."""
    lam = np.array([4.0, 1.0, 0.25])
    np.testing.assert_allclose(spectrum_eigvals(lam, 1.0), lam)
    flat = spectrum_eigvals(lam, 0.0)
    scale = 1.0 / np.sqrt(1.0 + 3.0 * flat)
    assert np.ptp(scale) == 0.0
    x = np.eye(3)
    np.testing.assert_allclose(whiten(x, np.eye(3), flat, 3.0), x * scale[0])


def test_oracle_eigvals_equal_pool_spectrum_on_pool_basis():
    """With the cohort basis equal to the pool basis, the oracle spectrum is the pool's."""
    basis = np.linalg.qr(np.random.default_rng(0).standard_normal((5, 3)))[0].T
    lam = np.array([3.0, 2.0, 1.0])
    np.testing.assert_allclose(oracle_eigvals(basis, _pool(basis, lam)), lam)


def test_oracle_slope_recovers_beta():
    """v = lambda ** beta gives slope exactly beta."""
    lam = np.array([8.0, 3.0, 1.0, 0.2])
    assert oracle_slope(lam, lam**0.6) == pytest.approx(0.6)


def test_within_patient_basis_is_orthonormal_and_rank_capped():
    """Rows are orthonormal and at most k."""
    x = np.random.default_rng(1).standard_normal((2 * 3 * 8, 6))
    basis = within_patient_basis(TrainingTable(x, np.zeros(len(x), int)), 2, 3, 4)
    assert basis.shape[0] <= 4
    np.testing.assert_allclose(basis @ basis.T, np.eye(len(basis)), atol=1e-10)


def test_select_beta_ties_go_to_larger_beta():
    """A tie keeps beta = 1; a strict improvement moves away from it."""
    tie = {"0.0": 0.8, "0.25": 0.8, "0.5": 0.8, "0.75": 0.8, "1.0": 0.8}
    assert select_beta(tie) == 1.0
    assert select_beta({**tie, "0.5": 0.9}) == 0.5


def _record(score: float, kappa: float) -> dict:
    """Minimal well-formed run record with one kappa score."""
    return {
        "dataset": {},
        "feature_extraction": {},
        "method": "logreg",
        "param": "lambda=1.0",
        "grid": {"g": 5, "m": 32, "draw": 0},
        "selected_lambda": 1.0,
        "solver": {},
        "arm": "x",
        "kappa_factor": kappa,
        "kappa_validation_scores": {str(kappa): score},
    }


def test_fit_bt_copies_rwc_record_when_beta_one_wins(tmp_path):
    """If no beta strictly beats RWc, Bt is RWc's record with beta = 1."""
    out, reused = tmp_path / "e22", tmp_path / "e17"
    config = {
        "paths": {"outputs": str(out)},
        "slurm": {"whitening_outputs": str(reused)},
    }
    paths = split_paths(ensure_dirs(config), 0)
    for family in BETAS:
        write_run_record(allocation_dir(paths, f"{family}5", 0), _record(0.7, 1.0))
    reused_paths = split_paths(ensure_dirs({"paths": {"outputs": str(reused)}}), 0)
    write_run_record(allocation_dir(reused_paths, "RWc5", 0), _record(0.8, 10.0))

    fit_mod._fit_bt(config, paths, 5, 0, 0)

    rec = read_run_record(allocation_dir(paths, "Bt5", 0), array_fields=())
    assert rec["arm"] == "Bt5" and rec["beta"] == 1.0
    assert rec["kappa_factor"] == 10.0
    assert "Bt5" in ARMS
