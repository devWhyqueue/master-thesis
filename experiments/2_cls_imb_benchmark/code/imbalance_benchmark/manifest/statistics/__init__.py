"""Manifest distribution statistics shared by natural and controlled conditions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

import numpy as np
import pandas as pd

ClassKey = TypeVar("ClassKey", str, int)


def normalized_entropy(counts: list[int]) -> float:
    """Compute the report's inverted normalized entropy, 1 - H/log(K)."""
    total = sum(counts)
    if len(counts) <= 1 or total <= 0:
        return 0.0
    probabilities = np.asarray(counts, dtype=float) / total
    probabilities = probabilities[probabilities > 0]
    return 1.0 - float(-(probabilities * np.log(probabilities)).sum()) / np.log(
        len(counts)
    )


def achieved_rho(counts: Mapping[ClassKey, int]) -> float:
    """Compute the realized head/tail imbalance ratio from per-class counts."""
    values = [count for count in counts.values() if count > 0]
    return max(values) / min(values) if values else 1.0


def _distribution_statistics(rows: pd.DataFrame, level: str) -> dict[str, Any]:
    frame = (
        rows if level == "patch" else rows.drop_duplicates(["slide_id", "cancer_type"])
    )
    counts = {
        str(name): int(count)
        for name, count in frame["cancer_type"].value_counts().items()
    }
    return {
        "counts": counts,
        "achieved_rho": achieved_rho(counts),
        "normalized_entropy": normalized_entropy(list(counts.values())),
    }


def support_statistics(rows: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Report the required realized patch- and slide-level distributions."""
    return {
        "patch": _distribution_statistics(rows, "patch"),
        "slide": _distribution_statistics(rows, "slide"),
    }
