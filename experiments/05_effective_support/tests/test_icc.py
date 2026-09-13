"""Unit tests for ICC estimation and effective support formulas."""

from __future__ import annotations

import numpy as np
import pytest

from breadth.icc import (
    cell_effective_support,
    compute_class_icc,
    design_effect,
    effective_support,
    pca_leading_direction,
)


def test_design_effect_formula():
    """DE_c(m) = 1 + (m - 1) * ICC_c."""
    assert design_effect(m=1, icc=0.2) == 1.0
    assert np.isclose(design_effect(m=8, icc=0.1), 1.0 + 7 * 0.1)
    assert np.isclose(design_effect(m=32, icc=0.05), 1.0 + 31 * 0.05)


def test_effective_support_formula_and_limit():
    """N_eff = Gm / DE and limit m->inf equals G / ICC."""
    g, m, icc = 10, 16, 0.2
    de = 1.0 + 15 * 0.2  # 4.0
    expected = (10 * 16) / de  # 160 / 4 = 40
    assert np.isclose(effective_support(g, m, icc), expected)

    # Large m approaches G / ICC
    eff_large = effective_support(g, m=100000, icc=icc)
    assert np.isclose(eff_large, g / icc, rtol=1e-3)


def test_cell_effective_support_average():
    """cell_effective_support averages N_eff,c over classes."""
    class_iccs = {"c1": 0.1, "c2": 0.2}
    g, m = 10, 16
    neff_1 = effective_support(g, m, 0.1)
    neff_2 = effective_support(g, m, 0.2)
    expected_mean = 0.5 * (neff_1 + neff_2)
    assert np.isclose(cell_effective_support(g, m, class_iccs), expected_mean)


def test_pca_leading_direction():
    """Leading direction is a unit vector aligned with the primary axis."""
    rng = np.random.default_rng(42)
    # Generate data with dominant variance along x-axis
    x = rng.normal(0, 10, size=(100, 1))
    y = rng.normal(0, 0.1, size=(100, 1))
    data = np.hstack([x, y])

    direction = pca_leading_direction(data)
    assert np.isclose(np.linalg.norm(direction), 1.0)
    # Absolute dot product with unit x-vector [1, 0] should be ~1
    assert np.isclose(abs(direction[0]), 1.0, atol=1e-2)


def test_compute_class_icc_clusters():
    """ICC is high when between-patient variance dominates."""
    rng = np.random.default_rng(123)
    n_cases = 10
    patches_per_case = 20
    dim = 8

    # Each case has distinct center, small within-case noise
    centers = rng.normal(0, 5.0, size=(n_cases, dim))
    feat_list = []
    case_list = []
    for i in range(n_cases):
        noise = rng.normal(0, 0.2, size=(patches_per_case, dim))
        feat_list.append(centers[i] + noise)
        case_list.extend([f"case_{i}"] * patches_per_case)

    features = np.vstack(feat_list)
    case_ids = np.asarray(case_list)
    direction = pca_leading_direction(features)

    icc_val = compute_class_icc(features, case_ids, direction, rng)
    assert 0.0 <= icc_val <= 1.0
    assert icc_val > 0.8  # Strong clustering

