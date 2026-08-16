"""Fast exact capacity checks for patient- and slide-capped selections."""

from __future__ import annotations

import logging
import time
from typing import cast

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _mil_capacity(patient_counts: np.ndarray, patient_cap: int) -> int:
    return int(np.minimum(patient_counts, patient_cap).sum())


def _patch_capacity(
    sizes: np.ndarray, offsets: np.ndarray, patient_cap: int, slide_cap: int
) -> int:
    per_patient = np.add.reduceat(np.minimum(sizes, slide_cap), offsets)
    return int(np.minimum(per_patient, patient_cap).sum())


def _class_feasible_counts(rows: pd.DataFrame, minimum: int, is_mil: bool) -> set[int]:
    slides = rows.drop_duplicates("slide_id")
    patient_counts = slides["case_id"].value_counts().to_numpy(dtype=int)
    slide_counts = rows.groupby(["case_id", "slide_id"], sort=True).size()
    sizes = slide_counts.to_numpy(dtype=int)
    codes = cast(pd.MultiIndex, slide_counts.index).codes[0]
    offsets = np.flatnonzero(np.diff(codes, prepend=-1))
    available = len(slides) if is_mil else len(rows)
    capacities: dict[tuple[int, int], int] = {}
    feasible = set()
    for count in range(minimum, available + 1):
        caps = (
            int(np.floor(count * 0.10)),
            int(np.floor(count * 0.05)) if not is_mil else 0,
        )
        if caps not in capacities:
            capacities[caps] = (
                _mil_capacity(patient_counts, caps[0])
                if is_mil
                else _patch_capacity(sizes, offsets, *caps)
            )
        if count <= capacities[caps]:
            feasible.add(count)
    return feasible


def feasible_selection_counts(
    train_df: pd.DataFrame, minimum: int, is_mil: bool
) -> dict[str, set[int]]:
    """Return exact selectable counts under the contribution caps by class."""
    result: dict[str, set[int]] = {}
    for class_name, rows in train_df.groupby("cancer_type", sort=False):
        start = time.perf_counter()
        logger.info(
            "freeze: feasible counts: class %s, %d units", class_name, len(rows)
        )
        result[str(class_name)] = _class_feasible_counts(rows, minimum, is_mil)
        logger.info(
            "freeze: feasible counts: class %s done in %.1fs",
            class_name,
            time.perf_counter() - start,
        )
    return result
