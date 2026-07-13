from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["tier_metrics"]


def tier_metrics(
    class_names: list[str], tiers: dict[str, str], stats: dict[str, np.ndarray]
) -> dict[str, dict[str, Any]]:
    """Average classwise discrimination and calibration values within each tier."""
    output = {}
    for tier in ("head", "body", "tail"):
        indices = [
            index for index, name in enumerate(class_names) if tiers.get(name) == tier
        ]
        if indices:
            output[tier] = _tier_values(indices, stats)
    return output


def _tier_values(indices: list[int], stats: dict[str, np.ndarray]) -> dict[str, Any]:
    """Summarize the classwise arrays selected for one head/body/tail tier."""
    return {
        "precision": float(np.mean(stats["precision"][indices])),
        "recall": float(np.mean(stats["recall"][indices])),
        "f1": float(np.mean(stats["f1"][indices])),
        "support": int(np.sum(stats["support"][indices])),
        "nll": float(np.nanmean(stats["nll"][indices])),
        "brier": float(np.nanmean(stats["brier"][indices])),
    }
