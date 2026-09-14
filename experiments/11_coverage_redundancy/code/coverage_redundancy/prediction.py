"""Split-intercept model fitting and the composition-cohort prediction."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from coverage_redundancy.models import design, fit_replicates, group_mean
from coverage_redundancy.rows import CompositionRows, GridRows

__all__ = ["FittedModels", "Prediction", "fit_models", "predict_composition"]


class FittedModels(NamedTuple):
    """The four split-intercept OLS fits, one design each, over 2000 replicates."""

    theta_a: np.ndarray
    res_a: np.ndarray
    theta_b: np.ndarray
    res_b: np.ndarray
    theta_c: np.ndarray
    res_c: np.ndarray
    theta_cm: np.ndarray
    res_cm: np.ndarray


class Prediction(NamedTuple):
    """Predicted vs. observed composition accuracy and their contrast errors."""

    delta_con_hat: np.ndarray
    delta_sel_hat: np.ndarray
    con_error: np.ndarray
    sel_error: np.ndarray
    a_ran_hat: np.ndarray
    a_ran_obs: np.ndarray


def fit_models(grid: GridRows) -> FittedModels:
    """Fit models (a), (b), (c), and (c) without log G, per bootstrap replicate."""
    theta_a, res_a = fit_replicates(
        design(grid.split, [grid.log_neff, grid.log_g]), grid.accuracy
    )
    theta_b, res_b = fit_replicates(
        design(grid.split, [grid.log_neff_omega, grid.log_g]), grid.accuracy
    )
    theta_c, res_c = fit_replicates(
        design(grid.split, [grid.log_neff_omega, grid.r, grid.log_g]), grid.accuracy
    )
    theta_cm, res_cm = fit_replicates(
        design(grid.split, [grid.log_neff_omega, grid.r]), grid.accuracy
    )
    return FittedModels(
        theta_a, res_a, theta_b, res_b, theta_c, res_c, theta_cm, res_cm
    )


def predict_composition(
    theta_cm: np.ndarray,
    comp: CompositionRows,
    accuracy_by_allocation: dict[str, np.ndarray],
) -> Prediction:
    """Apply model (c) without log G to the composition cohorts and score it."""
    xcm_comp = design(comp.split, [comp.log_neff_omega, comp.r])
    predicted = theta_cm[:, 0][np.newaxis, :] + xcm_comp @ theta_cm[:, 1:].T

    a_clu_hat = group_mean(predicted, comp.allocation, "clustered")
    a_ran_hat = group_mean(predicted, comp.allocation, "random")
    a_dis_hat = group_mean(predicted, comp.allocation, "dispersed")
    a_clu_obs = accuracy_by_allocation["clustered"].mean(axis=0)
    a_ran_obs = accuracy_by_allocation["random"].mean(axis=0)
    a_dis_obs = accuracy_by_allocation["dispersed"].mean(axis=0)

    delta_con_hat = a_ran_hat - a_clu_hat
    delta_sel_hat = a_dis_hat - a_ran_hat
    con_error = delta_con_hat - (a_ran_obs - a_clu_obs)
    sel_error = delta_sel_hat - (a_dis_obs - a_ran_obs)
    return Prediction(
        delta_con_hat, delta_sel_hat, con_error, sel_error, a_ran_hat, a_ran_obs
    )
