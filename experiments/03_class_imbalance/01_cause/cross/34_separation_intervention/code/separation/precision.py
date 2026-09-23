"""Gate 4 (precision): project the main experiment's power for one primary directional effect
from the pilot's own centred validation contrasts, resampling whole (split, draw) units so
simulated studies never treat patches or repeated patient appearances as independent.
"""

from __future__ import annotations

import numpy as np

from separation import (
    MAIN_DRAWS,
    N_SPLITS,
    PRECISION_EFFECT_PP,
    PRECISION_N_BOOTSTRAP,
    PRECISION_N_SIM,
    PRECISION_SEED,
)

__all__ = ["projected_power"]


def projected_power(
    per_split_contrast: np.ndarray,
    n_sim: int = PRECISION_N_SIM,
    n_bootstrap: int = PRECISION_N_BOOTSTRAP,
    seed: int = PRECISION_SEED,
    effect_pp: float = PRECISION_EFFECT_PP,
) -> float:
    """Fraction of simulated main-scale studies whose 95% CI excludes zero in the right direction.

    ``per_split_contrast`` is one observed contrast per split from the pilot (draw 0), oriented so
    a positive value is the hypothesised direction. Each simulated study draws ``len(MAIN_DRAWS)``
    synthetic per-draw observations per split from ``effect_pp`` plus a resampled centred pilot
    residual, then a nested percentile bootstrap (resampling whole synthetic draws) mimics this
    codebase's usual draw-resampling CI.
    """
    contrast = np.asarray(per_split_contrast, dtype=np.float64)
    residual = contrast - contrast.mean()
    n_draws = len(MAIN_DRAWS)
    n_obs = N_SPLITS * n_draws
    rng = np.random.default_rng(seed)

    studies = effect_pp + rng.choice(residual, size=(n_sim, n_obs))
    idx = rng.integers(0, n_obs, size=(n_sim, n_bootstrap, n_obs))
    boot_means = studies[np.arange(n_sim)[:, None, None], idx].mean(axis=2)
    lo = np.percentile(boot_means, 2.5, axis=1)
    return float((lo > 0.0).mean())
