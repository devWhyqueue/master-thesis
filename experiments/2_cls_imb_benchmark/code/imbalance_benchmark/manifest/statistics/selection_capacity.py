"""Fast exact capacity checks for patient- and slide-capped selections."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _mil_capacity(patient_counts: np.ndarray, patient_cap: int) -> int:
    return int(np.minimum(patient_counts, patient_cap).sum())


def _patch_capacity(
    patient_slides: list[np.ndarray], patient_cap: int, slide_cap: int
) -> int:
    return sum(
        min(patient_cap, int(np.minimum(slides, slide_cap).sum()))
        for slides in patient_slides
    )


def _class_feasible_counts(rows: pd.DataFrame, minimum: int, is_mil: bool) -> set[int]:
    slides = rows.drop_duplicates("slide_id")
    patient_counts = slides["case_id"].value_counts().to_numpy(dtype=int)
    slide_counts = rows.groupby(["case_id", "slide_id"]).size()
    patient_slides = [
        counts.to_numpy(dtype=int)
        for _, counts in slide_counts.groupby(level="case_id", sort=False)
    ]
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
                else _patch_capacity(patient_slides, *caps)
            )
        if count <= capacities[caps]:
            feasible.add(count)
    return feasible


def feasible_selection_counts(
    train_df: pd.DataFrame, minimum: int, is_mil: bool
) -> dict[str, set[int]]:
    """Return exact selectable counts under the contribution caps by class."""
    return {
        str(class_name): _class_feasible_counts(rows, minimum, is_mil)
        for class_name, rows in train_df.groupby("cancer_type", sort=False)
    }
