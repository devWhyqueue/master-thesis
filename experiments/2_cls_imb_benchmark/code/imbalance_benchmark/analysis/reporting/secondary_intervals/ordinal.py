from __future__ import annotations

import numpy as np

from imbalance_benchmark.analysis.inference.bootstrap import PatientWeights
from imbalance_benchmark.analysis.reporting.secondary_intervals.weighted import (
    weighted_mean,
)

__all__ = ["ordinal_metrics"]


def _quadratic_weighted_kappa(
    labels: np.ndarray,
    predictions: np.ndarray,
    weights: PatientWeights,
    n_classes: int,
) -> np.ndarray:
    observed = np.zeros((n_classes, n_classes, weights.n_replicates), dtype=float)
    for truth in range(n_classes):
        for predicted in range(n_classes):
            mask = (labels == truth) & (predictions == predicted)
            observed[truth, predicted] = weights.sums(1.0, mask)
    total = observed.sum(axis=(0, 1))
    expected = (
        observed.sum(axis=1)[:, None, :] * observed.sum(axis=0)[None, :, :]
    ) / np.maximum(total, 1e-12)
    scale = max(n_classes - 1, 1) ** 2
    disagreement = (
        np.arange(n_classes)[:, None] - np.arange(n_classes)[None, :]
    ) ** 2 / scale
    numerator = (observed * disagreement[:, :, None]).sum(axis=(0, 1))
    denominator = (expected * disagreement[:, :, None]).sum(axis=(0, 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denominator > 0, 1.0 - numerator / denominator, np.nan)


def ordinal_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    weights: PatientWeights,
    n_classes: int,
) -> dict[str, np.ndarray]:
    """PANDA-only ordinal endpoints: quadratic weighted kappa and mean absolute error."""
    return {
        "quadratic_weighted_kappa": _quadratic_weighted_kappa(
            labels, predictions, weights, n_classes
        ),
        "ordinal_mean_absolute_error": weighted_mean(
            np.abs(labels - predictions).astype(float), weights
        ),
    }
