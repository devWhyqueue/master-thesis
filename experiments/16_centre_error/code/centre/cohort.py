"""Nested random cohorts, independent swap cohorts, and their training features."""

from __future__ import annotations

import logging
from typing import NamedTuple, cast

import numpy as np
import pandas as pd

from breadth.sampling import (
    derive_draw_seed,
    eligible_patients_by_class,
    load_features_for_df,
    sample_patient_patches_round_robin,
)

from centre import COHORT_SEED, PATIENT_COUNTS, POOL_DEPTH, patches_per_patient

__all__ = ["Cohorts", "TrainingTable", "draw_cohorts", "patient_rows", "training_table"]

logger = logging.getLogger(__name__)

_SWAP_SIZE = 5


class Cohorts(NamedTuple):
    """Per class: the nested twenty-patient cohort and an independent five-patient swap cohort."""

    nested: list[list[str]]
    swap: list[list[str]]


class TrainingTable(NamedTuple):
    """Float64 features (n, d) and integer class targets (n,)."""

    x: np.ndarray
    y: np.ndarray


def draw_cohorts(
    train_df: pd.DataFrame, class_names: list[str], split_idx: int, draw_idx: int
) -> Cohorts:
    """Draw twenty eligible patients per class (the first 5 and 10 are the smaller cohorts) and a swap cohort."""
    g_max = max(PATIENT_COUNTS)
    nested, swap = [], []
    for ci, name in enumerate(class_names):
        eligible = eligible_patients_by_class(train_df, name, POOL_DEPTH)
        seed = derive_draw_seed(COHORT_SEED, split_idx, g_max, POOL_DEPTH, draw_idx, ci)
        rng = np.random.default_rng(seed)
        chosen = [str(p) for p in rng.choice(eligible, size=g_max, replace=False)]
        rest = [p for p in eligible if p not in set(chosen)]
        if len(rest) < _SWAP_SIZE:
            logger.warning("Class %s: swap cohort overlaps patients 6-20", name)
            rest = [p for p in eligible if p not in set(chosen[:_SWAP_SIZE])]
        nested.append(chosen)
        swap.append([str(p) for p in rng.choice(rest, size=_SWAP_SIZE, replace=False)])
    return Cohorts(nested, swap)


def patient_rows(class_df: pd.DataFrame, patients: list[str], m: int) -> list[int]:
    """Manifest row indices of the first ``m`` round-robin patches of each patient."""
    by_case = class_df.groupby(class_df["case_id"].astype(str))
    rows: list[int] = []
    for patient in patients:
        rows.extend(
            sample_patient_patches_round_robin(
                cast(pd.DataFrame, by_case.get_group(patient)), m
            )
        )
    return rows


def training_table(
    train_df: pd.DataFrame, class_names: list[str], cohorts: list[list[str]], g: int
) -> TrainingTable:
    """Features and targets of the first ``g`` patients per class at the budget's patch depth."""
    rows: list[int] = []
    y: list[int] = []
    for ci, name in enumerate(class_names):
        class_df = cast(pd.DataFrame, train_df[train_df["cancer_type"] == name])
        class_rows = patient_rows(class_df, cohorts[ci][:g], patches_per_patient(g))
        rows.extend(class_rows)
        y.extend([ci] * len(class_rows))
    x = load_features_for_df(train_df.loc[rows]).astype(np.float64)
    return TrainingTable(x, np.asarray(y, dtype=np.int64))
