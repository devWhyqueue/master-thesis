"""Slide-disjoint CAMELYON16 split helpers (case_id == slide_id)."""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

SPLITS = ("train", "validation", "test")


def split_cases(slide_frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Return stratified case-level split assignments keyed by case_id.

    ``slide_frame`` has one row per slide with ``case_id`` and ``slide_label``.
    Slides are stratified by their slide-level tumor/normal label so tumor and
    normal slides are spread across train/validation/test in ~70/15/15 shares.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict[str, str]] = []
    for _, group in slide_frame.groupby("slide_label", sort=False):
        rows.extend(_split_group(group, rng))
    return pd.DataFrame(rows)


def assert_slide_disjoint(frame: pd.DataFrame) -> None:
    """Raise if any case appears in more than one split."""
    split_counts = cast(pd.Series, frame.groupby("case_id")["split"].nunique())
    leaking = [str(case) for case, count in split_counts.items() if int(count) > 1]
    if leaking:
        raise ValueError(f"CAMELYON16 slide leakage: {leaking[:5]}")


def _split_group(group: pd.DataFrame, rng: np.random.Generator) -> list[dict[str, str]]:
    cases = group["case_id"].astype(str).to_numpy()
    rng.shuffle(cases)
    n_cases = len(cases)
    n_train = max(1, int(round(n_cases * 0.70)))
    n_val = max(1, int(round(n_cases * 0.15)))
    if n_train + n_val >= n_cases and n_cases > 1:
        n_val = max(0, n_cases - n_train - 1)
    return (
        [{"case_id": case, "split": "train"} for case in cases[:n_train]]
        + [
            {"case_id": case, "split": "validation"}
            for case in cases[n_train : n_train + n_val]
        ]
        + [{"case_id": case, "split": "test"} for case in cases[n_train + n_val :]]
    )
