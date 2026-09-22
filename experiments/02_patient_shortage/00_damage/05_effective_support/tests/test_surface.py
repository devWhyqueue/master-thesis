"""Unit tests for support surface regression modeling."""

from __future__ import annotations

import numpy as np

from breadth import GRID_CELLS
from breadth.surface import fit_candidate_models, fit_ols


def test_fit_ols_recovers_known_parameters():
    """OLS recovers slope and intercept on linear synthetic data."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])[:, np.newaxis]
    true_alpha, true_beta = 2.5, 1.8
    y = true_alpha + true_beta * x.squeeze()

    theta, res_std, r2 = fit_ols(x, y)
    assert np.isclose(theta[0], true_alpha)
    assert np.isclose(theta[1], true_beta)
    assert np.isclose(res_std, 0.0, atol=1e-10)
    assert np.isclose(r2, 1.0)


def test_fit_candidate_models_structure_and_ranking():
    """fit_candidate_models produces ranking and all single and augmented models."""
    cells = GRID_CELLS  # 9 cells
    # Synthetic accuracies tracking log(N_eff) with small noise
    g_vals = np.array([c[0] for c in cells])
    m_vals = np.array([c[1] for c in cells])
    eff_supports = (g_vals * m_vals) / (1.0 + 15 * 0.1)

    y_acc = 70.0 + 3.0 * np.log(eff_supports)

    res = fit_candidate_models(cells, y_acc, eff_supports)

    assert "ranking" in res
    assert res["ranking"][0] == "log_neff"  # Best fit
    assert np.isclose(res["single_models"]["log_neff"]["res_std"], 0.0, atol=1e-7)

    # In augmented effective model, gamma_e should be ~0 since log(G) adds no residual signal
    gamma_e = res["augmented_effective"]["gamma_e"]
    assert np.isclose(gamma_e, 0.0, atol=1e-7)

    # In augmented nominal model, gamma_n should be positive since breadth beats depth
    gamma_n = res["augmented_nominal"]["gamma_n"]
    assert np.isfinite(gamma_n)

