from __future__ import annotations

import numpy as np

from imbalance_benchmark.analysis.inference.bootstrap import PatientWeights

__all__ = ["weighted_mean"]


def weighted_mean(
    values: np.ndarray, weights: PatientWeights, mask: np.ndarray | None = None
) -> np.ndarray:
    """Row-weighted mean of ``values`` under ``weights``, restricted to ``mask``."""
    denominator = weights.sums(1.0, mask)
    numerator = weights.sums(values, mask)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denominator > 0, numerator / denominator, np.nan)
