"""Pre-registered descriptive sensitivities (a)/(b)/(c) for the primary total_100 pool.

Split out of ``classprops.analyze`` purely to stay under this codebase's per-file length limit;
the two modules are one analysis stage.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from prevalence import BALANCED

from classprops.model import DatasetCovariates, fit_model, fit_pool, gather_covariates
from classprops.pool import Observations, observations_from_arrays

__all__ = ["sensitivity_a", "sensitivity_b", "sensitivity_c"]


def sensitivity_a(
    tcga_obs: Observations,
    tcga_cov: DatasetCovariates,
    bracs_obs: Observations,
    bracs_cov: DatasetCovariates,
) -> dict[str, Any]:
    """(a) h-only / m-only M1 variants, plus the h-m correlation and its VIF."""
    m_h = fit_model(tcga_obs, tcga_cov, bracs_obs, bracs_cov, covariates=("h",))
    m_m = fit_model(tcga_obs, tcga_cov, bracs_obs, bracs_cov, covariates=("m",))
    h_point = np.concatenate([tcga_cov.h[:, 0], bracs_cov.h[:, 0]])
    m_point = np.concatenate([tcga_cov.m[:, 0], bracs_cov.m[:, 0]])
    r = float(np.corrcoef(h_point, m_point)[0, 1])
    return {
        "h_only_G": m_h.g.tolist(),
        "m_only_G": m_m.g.tolist(),
        "h_m_correlation": r,
        "h_m_vif": 1.0 / max(1.0 - r * r, 1e-9),
    }


def sensitivity_b(
    tcga_obs: Observations,
    tcga_cov: DatasetCovariates,
    bracs_obs: Observations,
    bracs_cov: DatasetCovariates,
) -> dict[str, Any]:
    """(b) absolute allocation basis (tail patches/patient) instead of relative-to-balanced."""
    shift = math.log2(BALANCED)
    tcga_abs = tcga_obs._replace(z=tcga_obs.z + shift)
    bracs_abs = bracs_obs._replace(z=bracs_obs.z + shift)
    return fit_pool(tcga_abs, tcga_cov, bracs_abs, bracs_cov)


def sensitivity_c(
    tcga_arrays: dict[str, np.ndarray],
    bracs_obs: Observations,
    bracs_cov: DatasetCovariates,
) -> dict[str, Any] | None:
    """(c) drop exp-30/32 fixed orders on TCGA-UT: random r100 draws only."""
    if "sens_c_100_delta" not in tcga_arrays:
        return None
    tcga_obs = observations_from_arrays(tcga_arrays, "sens_c_100")
    tcga_cov = gather_covariates(tcga_obs, tcga_arrays["h"], tcga_arrays["m"])
    return fit_pool(tcga_obs, tcga_cov, bracs_obs, bracs_cov)
