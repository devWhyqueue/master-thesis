"""Support-surface refits: vary accuracies (test-patient) or correlation (training-patient)."""

from __future__ import annotations

import logging
import math
from typing import Any, Callable

import numpy as np

from breadth import GRID_CELLS, N_REPLICATES
from breadth.analyze.secondary import pack_estimate
from breadth.surface import fit_candidate_models

__all__ = ["LN2", "fit_all_surfaces", "paired_differences"]

logger = logging.getLogger(__name__)

LN2 = math.log(2.0)


def _replicate_fits(
    n_replicates: int,
    accs_at: Callable[[int], np.ndarray],
    neff_at: Callable[[int], np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit the support surface once per replicate; return gamma_e, res_std, beta."""
    gamma_e = np.empty(n_replicates)
    res_std_neff = np.empty(n_replicates)
    beta = np.empty(n_replicates)
    for i in range(n_replicates):
        fit = fit_candidate_models(GRID_CELLS, accs_at(i), neff_at(i))
        gamma_e[i] = fit["augmented_effective"]["gamma_e"]
        res_std_neff[i] = fit["single_models"]["log_neff"]["res_std"]
        beta[i] = fit["single_models"]["log_neff"]["beta"]
    return gamma_e, res_std_neff, beta


def _surface_estimate(test_dist: np.ndarray, train_dist: np.ndarray) -> dict[str, Any]:
    test_est, train_est = pack_estimate(test_dist), pack_estimate(train_dist)
    return {
        "point": test_est["point"],
        "test_ci": [test_est["ci_2_5"], test_est["ci_97_5"]],
        "training_ci": [train_est["ci_2_5"], train_est["ci_97_5"]],
    }


def _measure_surface(
    accs_matrix: np.ndarray,
    accs_point: np.ndarray,
    neff_point: np.ndarray,
    neff_all: np.ndarray,
) -> tuple[dict[str, Any], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Test-patient (vary accuracies) and training-patient (vary correlation) fits."""
    test_gamma_e, test_res, test_beta = _replicate_fits(
        N_REPLICATES, lambda i: accs_matrix[i], lambda i: neff_point
    )
    train_gamma_e, train_res, train_beta = _replicate_fits(
        N_REPLICATES, lambda i: accs_point, lambda i: neff_all[i]
    )
    fit = {
        "b": _surface_estimate(test_gamma_e * LN2, train_gamma_e * LN2),
        "beta": _surface_estimate(test_beta, train_beta),
        "res_std_neff": _surface_estimate(test_res, train_res),
    }
    return fit, (test_gamma_e * LN2, train_gamma_e * LN2, test_res, train_res)


def fit_all_surfaces(
    accs_matrix: np.ndarray,
    accs_point: np.ndarray,
    neff_point: dict[str, np.ndarray],
    neff_all: dict[str, np.ndarray],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Fit both measures' surfaces; return surfaces plus their paired b/res_std dists."""
    surfaces: dict[str, Any] = {}
    b_dists: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    res_dists: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for measure, measure_neff_point in neff_point.items():
        logger.info("Fitting support surface for the %s measure...", measure)
        fit, (test_b, train_b, test_res, train_res) = _measure_surface(
            accs_matrix, accs_point, measure_neff_point, neff_all[measure]
        )
        surfaces[measure] = fit
        b_dists[measure] = (test_b, train_b)
        res_dists[measure] = (test_res, train_res)
    return surfaces, b_dists, res_dists


def paired_differences(
    b_dists: dict[str, tuple[np.ndarray, np.ndarray]],
    res_dists: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    """Full-minus-single differences, paired by shared replicate index."""
    diff_test_b = b_dists["full"][0] - b_dists["single"][0]
    diff_train_b = b_dists["full"][1] - b_dists["single"][1]
    diff_test_res = res_dists["full"][0] - res_dists["single"][0]
    diff_train_res = res_dists["full"][1] - res_dists["single"][1]
    return {
        "b": _surface_estimate(diff_test_b, diff_train_b),
        "res_std_neff": _surface_estimate(diff_test_res, diff_train_res),
    }
