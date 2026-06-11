"""Parse and validate tuning parameter payloads."""

from __future__ import annotations

import json
from typing import Any

from tcga_ut_imbalanced.evaluation.tuning_grid import validate_tuning_params


def parse_tuning_params(
    benchmark: str, method: str, raw: str | None
) -> dict[str, float]:
    """Parse JSON tuning parameters for one benchmark method."""
    if raw is None:
        return {}
    payload: Any = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("--tuning-params must be a JSON object")
    validate_tuning_params(benchmark, method, payload)
    return {str(key): float(value) for key, value in payload.items()}
