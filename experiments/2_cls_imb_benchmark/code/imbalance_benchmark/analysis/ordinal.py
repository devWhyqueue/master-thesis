"""Ordinal-only endpoints for PANDA ISUP grading."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import cohen_kappa_score, mean_absolute_error


def ordinal_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    """Return quadratic-weighted kappa and ordinal mean absolute error."""
    kappa = cohen_kappa_score(labels, predictions, weights="quadratic")
    return {
        "quadratic_weighted_kappa": float(kappa) if len(np.unique(labels)) > 1 else float("nan"),
        "ordinal_mean_absolute_error": float(mean_absolute_error(labels, predictions)),
    }
