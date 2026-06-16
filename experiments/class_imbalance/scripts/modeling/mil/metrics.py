from __future__ import annotations

import numpy as np

from common_code.metrics.calibration import calibration_metrics as _calibration_metrics
from common_code.metrics.payload import _tier_metrics


def extra_metrics(
    y_true: list[int],
    probabilities: list[list[float]],
    precision: np.ndarray,
    recall: np.ndarray,
    f1: np.ndarray,
    support: np.ndarray,
    n_classes: int,
    tier_support: np.ndarray | None = None,
) -> dict[str, object]:
    """Compute extra classification metrics including support tiers and calibration."""
    resolved = tier_support if tier_support is not None else support
    payload: dict[str, object] = {
        "support_tier_metrics": _tier_metrics(precision, recall, f1, support, resolved)
    }
    payload.update(_calibration_metrics(y_true, probabilities, n_classes))
    return payload


__all__ = ["_calibration_metrics", "_tier_metrics", "extra_metrics"]
