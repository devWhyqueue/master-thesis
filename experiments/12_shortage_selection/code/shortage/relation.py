"""Secondary analysis 4: apply exp-11's fitted coverage/similarity relation to exp-12."""

from __future__ import annotations

from typing import Any

import numpy as np
from imbalance_benchmark.common import ensure_dirs, split_paths

from breadth import N_SPLITS
from breadth.analyze.secondary import pack_estimate

from coverage_redundancy.census import load_quantities
from coverage_redundancy.models import _arm_descriptive, design, group_mean
from coverage_redundancy.prediction import fit_models
from coverage_redundancy.rows import cohort_index, grid_rows

from composition.analyze import Context

from shortage import ALLOCATIONS, N_DRAWS, exp5_config, exp11_config

__all__ = ["relation_secondary"]


def _shortage_rows(
    quantities: dict[str, Any], n_splits: int, n_draws: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split index and [log_neff_omega, r] predictor rows for every exp-12 cohort."""
    index = cohort_index(quantities)
    splits, rows, allocation = [], [], []
    for name in ALLOCATIONS:
        for s in range(n_splits):
            for d in range(n_draws):
                record = index[("shortage", s, 5, 32, name, d)]
                splits.append(s)
                rows.append([record["log_neff_omega"], record["r"]])
                allocation.append(name)
    predictors = np.array(rows)
    return np.array(splits), predictors[:, 0], predictors[:, 1], np.array(allocation)


def _predicted_arms(
    theta_cm: np.ndarray, quantities: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, float, float, float, float]:
    """Both arms' predicted accuracy, and their mean r / log_neff_omega predictors."""
    split, log_neff_omega, r, allocation = _shortage_rows(quantities, N_SPLITS, N_DRAWS)
    xcm = design(split, [log_neff_omega, r])
    predicted = theta_cm[:, 0][np.newaxis, :] + xcm @ theta_cm[:, 1:].T
    a_ran_hat = group_mean(predicted, allocation, "random")
    a_dis_hat = group_mean(predicted, allocation, "dispersed")
    r_ran, r_dis = r[allocation == "random"].mean(), r[allocation == "dispersed"].mean()
    lno_ran = log_neff_omega[allocation == "random"].mean()
    lno_dis = log_neff_omega[allocation == "dispersed"].mean()
    return (
        a_ran_hat,
        a_dis_hat,
        float(r_ran),
        float(r_dis),
        float(lno_ran),
        float(lno_dis),
    )


def _fit_theta_cm(config: dict[str, Any], ctx: Context) -> np.ndarray:
    """Refit exp-11's model (c) without log G on its own random-cohort grid."""
    index11 = cohort_index(load_quantities(exp11_config(config)))
    paths5 = {
        s: split_paths(ensure_dirs(exp5_config(config)), s) for s in range(N_SPLITS)
    }
    grid = grid_rows(
        config, paths5, ctx.ctx_l, ctx.perms, len(ctx.class_names), index11
    )
    return fit_models(grid).theta_cm


def _arms_descriptives(quantities: dict[str, Any]) -> dict[str, Any]:
    """Mean omega and Neff^omega of both exp-12 arms."""
    return {
        name: _arm_descriptive(
            [
                c
                for c in quantities["cohorts"]
                if c["source"] == "shortage" and c["allocation"] == name
            ]
        )
        for name in ALLOCATIONS
    }


def relation_secondary(
    config: dict[str, Any],
    quantities: dict[str, Any],
    delta_sel_obs: np.ndarray,
    ctx: Context,
) -> dict[str, Any]:
    """Apply exp-11's fitted relation (model c minus log G) to the two exp-12 arms."""
    theta_cm = _fit_theta_cm(config, ctx)

    a_ran_hat, a_dis_hat, r_ran, r_dis, lno_ran, lno_dis = _predicted_arms(
        theta_cm, quantities
    )
    delta_sel_hat = a_dis_hat - a_ran_hat
    coverage_term = theta_cm[:, 4] * (r_dis - r_ran)
    similarity_term = theta_cm[:, 3] * (lno_dis - lno_ran)

    return {
        "a_ran_hat": pack_estimate(a_ran_hat),
        "a_dis_hat": pack_estimate(a_dis_hat),
        "delta_sel_hat": pack_estimate(delta_sel_hat),
        "sel_error": pack_estimate(delta_sel_hat - delta_sel_obs),
        "coverage_term": pack_estimate(coverage_term),
        "similarity_term": pack_estimate(similarity_term),
        "arms": _arms_descriptives(quantities),
    }
