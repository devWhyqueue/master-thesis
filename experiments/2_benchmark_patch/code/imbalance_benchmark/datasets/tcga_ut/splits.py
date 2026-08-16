"""TCGA-UT participant identity and participant-disjoint split assignment.

Shared by both the WSI (pre-extracted tensor) and patch (image-backed)
regimes: whichever evidence source a manifest is built from, participants
must still be assigned to exactly one split.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

__all__ = [
    "tcga_case_id",
    "assign_class_splits",
    "split_cases",
    "assert_case_disjoint",
]


def tcga_case_id(slide_id: str) -> str:
    """Return the participant barcode encoded in a TCGA slide identifier."""
    parts = slide_id.split("-")
    if len(parts) >= 3 and parts[0] == "TCGA":
        return "-".join(parts[:3])
    return slide_id


def assign_class_splits(
    units: list[str], seed: int, val_fraction: float = 0.15, test_fraction: float = 0.15
) -> dict[str, str]:
    """Assign per-class split units (participants) to train/validation/test."""
    rng = np.random.default_rng(seed)
    shuffled = list(units)
    rng.shuffle(shuffled)
    n_units = len(shuffled)
    if n_units == 1:
        return {shuffled[0]: "train"}
    if n_units == 2:
        return {shuffled[0]: "train", shuffled[1]: "test"}
    n_test = max(1, int(round(n_units * test_fraction)))
    n_val = max(1, int(round(n_units * val_fraction)))
    if n_test + n_val >= n_units:
        n_test, n_val = 1, 1
    assignments: dict[str, str] = {}
    for case_id in shuffled[:n_test]:
        assignments[case_id] = "test"
    for case_id in shuffled[n_test : n_test + n_val]:
        assignments[case_id] = "validation"
    for case_id in shuffled[n_test + n_val :]:
        assignments[case_id] = "train"
    return assignments


def split_cases(slide_manifest: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Return participant-disjoint split assignments keyed by case_id, per class."""
    rows: list[dict[str, str]] = []
    for _, class_df in slide_manifest.groupby("cancer_type"):
        assignments = assign_class_splits(
            sorted(class_df["case_id"].astype(str).unique()), seed
        )
        rows.extend(
            {"case_id": case_id, "split": split}
            for case_id, split in assignments.items()
        )
    return pd.DataFrame(rows)


def assert_case_disjoint(frame: pd.DataFrame) -> None:
    """Raise if any participant appears in more than one split."""
    split_counts = cast(pd.Series, frame.groupby("case_id")["split"].nunique())
    leaking = [
        str(case_id) for case_id, count in split_counts.items() if int(count) > 1
    ]
    if leaking:
        raise ValueError(f"TCGA-UT participant leakage: {leaking[:5]}")
