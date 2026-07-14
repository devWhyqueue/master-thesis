from __future__ import annotations

import numpy as np


def method_floor(patient_equals_slide: bool) -> dict[str, int]:
    """Return the fixed method-floor independent-unit minimums per class."""
    return {"slides": 20} if patient_equals_slide else {"patients": 10, "slides": 20}


def meets_method_floor(support: dict[str, int], patient_equals_slide: bool) -> bool:
    """Require every independent-unit floor applicable to the regime."""
    return all(
        support.get(unit, 0) >= minimum
        for unit, minimum in method_floor(patient_equals_slide).items()
    )


def stability_floor_from_curve(
    levels: list[int], ba: dict[int, list[float]], rcs: dict[int, list[list[float]]]
) -> int:
    """Return the first support level whose aggregate and classwise gains are stable."""
    mean_ba = np.mean(np.stack(list(ba.values())), axis=0)
    for idx in range(len(levels) - 1):
        gain = abs(float(mean_ba[idx + 1] - mean_ba[idx]))
        class_gain = max(
            abs(recalls[idx + 1][class_index] - recalls[idx][class_index])
            for recalls in rcs.values()
            for class_index in range(len(recalls[idx]))
        )
        if gain < 0.01 and class_gain < 0.02:
            return levels[idx]
    return levels[-1]
