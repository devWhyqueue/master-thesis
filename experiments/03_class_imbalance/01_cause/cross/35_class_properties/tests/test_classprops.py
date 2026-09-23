"""Unit tests for exp-35: cross-fitting excludes its own split, and the pooled model recovers a
known dataset-vs-covariate confound (shrinks to 0 when the BRACS/TCGA-UT gap is a function of h).
"""

from __future__ import annotations

import numpy as np
import pytest

from centre import N_DRAWS, N_SPLITS

from classprops.covariates import cross_fitted_headroom
from classprops.model import DatasetCovariates, fit_model, shrink_share
from classprops.pool import Observations


def test_cross_fitted_headroom_excludes_its_own_split() -> None:
    """Split s's headroom never changes when only split s's own fits are perturbed."""
    rng = np.random.default_rng(0)
    n_classes, n_replicates = 4, 3
    n_fits = N_SPLITS * N_DRAWS
    r1_stack = rng.normal(size=(n_fits, n_classes, n_replicates))
    w = np.ones((n_fits, n_replicates))
    names = [f"c{i}" for i in range(n_classes)]

    fit_split = np.repeat(np.arange(N_SPLITS), N_DRAWS)
    perturbed = r1_stack.copy()
    perturbed[fit_split == 0] += 1000.0  # only split 0's own fits change

    h_before = cross_fitted_headroom(r1_stack, w, names)
    h_after = cross_fitted_headroom(perturbed, w, names)

    np.testing.assert_allclose(h_after[0], h_before[0])  # split 0 excludes its own fits
    assert not np.allclose(h_after[1], h_before[1])  # split 1 pools split 0's (changed) fits
    assert not np.allclose(h_after[2], h_before[2])


def _dataset_observations(
    rng: np.random.Generator, mean_h: float, n: int, n_replicates: int
) -> tuple[Observations, DatasetCovariates]:
    """Synthetic (z, h) with delta generated as a deterministic function of both, no B term."""
    z = rng.normal(size=n) * 0.1
    h = rng.normal(loc=mean_h, scale=5.0, size=n)
    delta_point = 1.0 + 0.05 * z + 0.5 * h
    obs = Observations(
        delta=np.broadcast_to(delta_point[:, None], (n, n_replicates)).copy(),
        z=z,
        split_idx=np.zeros(n, dtype=np.int64),
        class_idx=np.arange(n) % 5,
        draw_weight=np.ones((n, n_replicates)),
    )
    cov = DatasetCovariates(
        h=np.broadcast_to(h[:, None], (n, n_replicates)).copy(),
        m=np.zeros((n, n_replicates)),
    )
    return obs, cov


def test_shrink_share_recovers_known_confound() -> None:
    """G1 shrinks to ~0 (shrink share -> 1) once h fully explains the dataset gap."""
    rng = np.random.default_rng(1)
    tcga_obs, tcga_cov = _dataset_observations(rng, mean_h=60.0, n=200, n_replicates=2)
    bracs_obs, bracs_cov = _dataset_observations(rng, mean_h=30.0, n=200, n_replicates=2)

    m0 = fit_model(tcga_obs, tcga_cov, bracs_obs, bracs_cov, covariates=())
    m1 = fit_model(tcga_obs, tcga_cov, bracs_obs, bracs_cov, covariates=("h", "m"))
    shrink = shrink_share(m0.g, m1.g)

    assert abs(m0.g[0]) > 1.0  # M0 (no h) sees a real, confounded dataset gap
    assert abs(m1.g[0]) < 1e-6  # M1 fully explains it via h, no direct B effect
    assert shrink[0] == pytest.approx(1.0, abs=1e-6)
