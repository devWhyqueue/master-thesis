"""Compute the entropy-based class-imbalance measure for each dataset.

Reads per-class counts from ``outputs/counts/counts.json`` and logs, for each
dataset, the tile-level and slide-level entropy-based imbalance ``1 - H_norm``
quoted in the report tables. This keeps those numbers reproducible from the
committed counts.

Imbalance measure: ``1 - H_norm``, with ``H_norm = H(p) / log(K)`` the Shannon
entropy of the class distribution normalised by its maximum ``log(K)``.
0 = perfectly balanced, ->1 = strongly imbalanced.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

COUNTS = Path(__file__).resolve().parent / "outputs" / "counts" / "counts.json"


def entropy_imbalance(counts: list[int]) -> float:
    """Return 1 - normalised entropy of the class-count distribution."""
    c = np.asarray([v for v in counts if v > 0], dtype=float)
    k = c.size
    if k <= 1:
        return 1.0
    p = c / c.sum()
    return float(1.0 - (-(p * np.log(p)).sum()) / np.log(k))


def head_tail_ratio(counts: list[int]) -> float:
    """Return the max:min (head-to-tail) class-support ratio."""
    positive = [v for v in counts if v > 0]
    return max(positive) / min(positive)


def main() -> None:
    """Log tile- and slide-level K, ratio, and 1 - H_norm for each dataset."""
    rows = json.loads(COUNTS.read_text(encoding="utf-8"))
    logger.info(
        "%-12s %-8s %-22s %3s %10s %8s",
        "dataset",
        "level",
        "label",
        "K",
        "head:tail",
        "1-Hnorm",
    )
    for row in rows:
        for level in ("tile", "slide"):
            values = list(row[level]["counts"].values())
            k = len([v for v in values if v > 0])
            logger.info(
                "%-12s %-8s %-22s %3d %8.1f:1 %8.3f",
                row["dataset"],
                level,
                row[level]["level"],
                k,
                head_tail_ratio(values),
                entropy_imbalance(values),
            )


def _self_check() -> None:
    """Sanity-check the measure on known distributions."""
    assert abs(entropy_imbalance([10, 10, 10, 10])) < 1e-9, "uniform -> 0"
    assert entropy_imbalance([100, 0, 0]) == 1.0, "single class -> 1"
    assert 0.0 < entropy_imbalance([90, 10]) < 1.0, "skewed binary in (0,1)"
    assert entropy_imbalance([99, 1]) > entropy_imbalance([60, 40])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _self_check()
    main()
