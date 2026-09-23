"""Gate 6 (precision): project the main design's 95% CI half-width for the primary BRACS joint
rescue from the pilot's own per-(split, draw) contrasts, resampling whole units so patches and
repeated patient appearances are never treated as independent (mirrors ``separation.precision``,
targeting a half-width bound instead of power against zero -- PLAN.md gate 6 asks for precision,
not just direction).
"""

from __future__ import annotations

import numpy as np

from joint import (
    MAIN_DRAWS,
    N_SPLITS,
    PRECISION_N_BOOTSTRAP,
    PRECISION_N_SIM,
    PRECISION_SEED,
)

__all__ = ["projected_halfwidth"]


def projected_halfwidth(
    per_obs_contrast: np.ndarray,
    n_sim: int = PRECISION_N_SIM,
    n_bootstrap: int = PRECISION_N_BOOTSTRAP,
    seed: int = PRECISION_SEED,
) -> float:
    """Median simulated main-scale study's 95% percentile-bootstrap CI half-width, in pp.

    ``per_obs_contrast`` is one observed rescue contrast per pilot (split, draw) unit. Each
    simulated main-scale study draws ``N_SPLITS * len(MAIN_DRAWS)`` synthetic observations from
    the pilot's own mean plus a resampled centred residual, then a nested percentile bootstrap
    (resampling whole synthetic observations) mimics this codebase's usual draw-resampling CI.
    """
    contrast = np.asarray(per_obs_contrast, dtype=np.float64)
    residual = contrast - contrast.mean()
    n_obs = N_SPLITS * len(MAIN_DRAWS)
    rng = np.random.default_rng(seed)

    studies = contrast.mean() + rng.choice(residual, size=(n_sim, n_obs))
    idx = rng.integers(0, n_obs, size=(n_sim, n_bootstrap, n_obs))
    boot_means = studies[np.arange(n_sim)[:, None, None], idx].mean(axis=2)
    lo = np.percentile(boot_means, 2.5, axis=1)
    hi = np.percentile(boot_means, 97.5, axis=1)
    return float(np.median((hi - lo) / 2.0))
