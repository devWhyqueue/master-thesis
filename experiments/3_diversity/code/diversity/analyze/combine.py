"""Equal-weight cross-split combination (see analyze/__init__.py's module docstring)."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats

from imbalance_benchmark.analysis.inference.gates import confidence_interval

__all__ = ["fisher_combine", "combine_splits"]


def fisher_combine(p_values: list[float]) -> float:
    """Fisher's method: -2 * sum(ln p_i) ~ chi2(2k) under the joint null."""
    clipped = np.clip(np.asarray(p_values, dtype=float), 1e-300, 1.0)
    statistic = float(-2.0 * np.log(clipped).sum())
    return float(stats.chi2.sf(statistic, df=2 * len(clipped)))


def combine_splits(
    distributions: list[np.ndarray], p_values: list[float]
) -> dict[str, Any]:
    """Equal-weight combination across the three patient splits.

    Keeps exp-2's replicate-0-is-observed convention: the combined array's
    index 0 is the true equal-weight point estimate, and the remaining
    entries are that point plus every split's own (recentred) bootstrap
    deviation pooled together, so ``confidence_interval`` and
    ``_recovery_comparison`` (both index-0-is-observed) work unchanged.
    """
    points = np.array([float(dist[0]) for dist in distributions])
    combined_point = float(points.mean())
    deviations = np.concatenate(
        [dist[1:] - dist[0] for dist in distributions if len(dist) > 1]
    )
    bootstrap_effect = np.concatenate([[combined_point], combined_point + deviations])
    return {
        "effect": combined_point,
        "ci": confidence_interval(bootstrap_effect),
        "p_value": fisher_combine(p_values),
        "bootstrap_effect": bootstrap_effect.tolist(),
    }
