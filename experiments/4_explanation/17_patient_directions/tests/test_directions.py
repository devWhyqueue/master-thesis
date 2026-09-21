"""Unit tests for the cohort eigenbasis, derived shares/interaction, and arm names."""

from __future__ import annotations

import numpy as np
import pytest

from centre import PATIENT_COUNTS
from centre.cohort import TrainingTable
from centre.fit import split_arm

from directions import ARMS, REUSED_ARMS
from directions.analyze import derived
from directions.basis import cohort_eigenbasis
from directions.fit import select_kappa


def test_cohort_eigenbasis_diagonalizes_explicit_pooled_covariance():
    """cohort_eigenbasis's eigenvectors/eigenvalues diagonalize the pooled between-patient covariance."""
    rng = np.random.default_rng(5)
    n_classes, g, m, d = 3, 4, 2, 6
    patient_means = rng.normal(size=(n_classes, g, d))
    x = np.repeat(patient_means, m, axis=1).reshape(n_classes * g * m, d)
    table = TrainingTable(x, np.repeat(np.arange(n_classes), g * m))

    basis, eigvals = cohort_eigenbasis(table, n_classes, g)

    dev = (patient_means - patient_means.mean(axis=1, keepdims=True)).reshape(-1, d)
    cov = dev.T @ dev / (n_classes * (g - 1))
    expected_vals = np.linalg.eigvalsh(cov)[::-1][: len(eigvals)]
    np.testing.assert_allclose(eigvals, expected_vals, atol=1e-10)
    np.testing.assert_allclose(basis @ cov @ basis.T, np.diag(eigvals), atol=1e-10)
    np.testing.assert_allclose(basis @ basis.T, np.eye(len(eigvals)), atol=1e-10)


def test_shares_and_interaction_follow_toy_gaps():
    """Shares, interaction, and gains match hand-computed values on toy arm distributions."""
    base = {
        "R": (60.0, 66.0, 70.0),
        "C": (70.0, 73.0, 75.0),
        "CW": (72.0, 74.0, 75.0),
        "RW": (65.0, 69.0, 71.0),
        "RWc": (64.0, 68.0, 70.0),
    }
    arm = {
        f"{f}{g}": np.array([v])
        for f, vals in base.items()
        for g, v in zip(PATIENT_COUNTS, vals)
    }

    out = derived(arm)

    assert out["share_C_5_to_10"][0] == pytest.approx(0.5)
    assert out["share_CW_5_to_10"][0] == pytest.approx(2.0 / 3.0)
    assert out["share_W_5_to_10"][0] == pytest.approx(1.0 / 3.0)
    assert out["share_Wc_5_to_10"][0] == pytest.approx(1.0 / 3.0)
    assert out["share_interaction_5_to_10"][0] == pytest.approx(
        2.0 / 3.0 - 0.5 - 1.0 / 3.0
    )
    assert out["interaction_5"][0] == pytest.approx(-3.0)
    assert out["RW5_minus_R5"][0] == pytest.approx(5.0)
    assert out["RWc5_minus_R5"][0] == pytest.approx(4.0)
    assert out["CW5_minus_C5"][0] == pytest.approx(2.0)


def test_arm_names_parse():
    """ARMS and REUSED_ARMS decode to their own (family, G) via centre's split_arm."""
    for arm in (*ARMS, *REUSED_ARMS):
        family, g = split_arm(arm)
        assert g in PATIENT_COUNTS
        assert arm == f"{family}{g}"
    assert split_arm("RWc10") == ("RWc", 10)


def test_select_kappa_prefers_larger_factor_within_tolerance():
    """select_kappa picks the best validation score and breaks ties towards the larger kappa."""
    assert select_kappa({"1.0": 0.7, "10.0": 0.6, "100.0": 0.5, "0.1": 0.8}) == "0.1"
    assert select_kappa({"0.1": 0.7, "1.0": 0.7, "10.0": 0.6}) == "1.0"
    assert select_kappa({"1.0": 0.6, "0.1": 0.5, "10.0": 0.65}) == "10.0"
