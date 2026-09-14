"""Random (anchor) and dispersed patient census rows at five patients; patch frames.

The random arm and the dispersed arm are not drawn here: they are read from
exp-10's own signed allocations (its stored five-patient seeds, and the first
five patients of its twenty-patient dispersed allocation, which equal the
five-patient dispersed allocation of this report because facility-location
selection is prefix-consistent). This module only computes this experiment's
own census rows from those patient lists, and materialises their patches.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd

from breadth.analyze.diagnostics import tissue_source_site

from neighbours.allocation import _patch_rows
from neighbours.coverage import (
    _coverage_distance,
    _embedding_matrix,
    _patients_of,
    _site_share,
)

from composition.allocation import _spread, _test_improvement

from shortage import ALLOCATIONS

__all__ = ["class_draw", "allocation_frames"]

_BROAD_M = 32
_N_DISPERSED = 5


def _coverage_r_values(
    embeddings: dict[tuple[str, str], np.ndarray],
    full_df: pd.DataFrame,
    c_name: str,
    random_alloc: list[str],
    dispersed: list[str],
    ran20: list[str],
) -> dict[str, float]:
    """Coverage distances of the random and dispersed arms, and of exp-10's r_20 (Eq. coverage)."""
    e_val = _embedding_matrix(
        embeddings, _patients_of(full_df, "validation", c_name), c_name
    )
    return {
        "r_5": _coverage_distance(
            e_val, _embedding_matrix(embeddings, random_alloc, c_name)
        ),
        "r_dispersed": _coverage_distance(
            e_val, _embedding_matrix(embeddings, dispersed, c_name)
        ),
        "r_ran20": _coverage_distance(
            e_val, _embedding_matrix(embeddings, ran20, c_name)
        ),
    }


def _site_share_non_focal(focal: str, patients: list[str]) -> float:
    """Share of an allocation's non-focal patients from the focal patient's site."""
    focal_site = {tissue_source_site(focal)}
    return _site_share([p for p in patients if p != focal], focal_site)


def _spread_and_site(
    embeddings: dict[tuple[str, str], np.ndarray],
    c_name: str,
    focal: str,
    random_alloc: list[str],
    dispersed: list[str],
) -> dict[str, float]:
    """Within-cohort spread and focal-site share of both arms."""
    return {
        "spread_random": _spread(embeddings, c_name, random_alloc),
        "spread_dispersed": _spread(embeddings, c_name, dispersed),
        "site_random": _site_share_non_focal(focal, random_alloc),
        "site_dispersed": _site_share_non_focal(focal, dispersed),
    }


def class_draw(
    embeddings: dict[tuple[str, str], np.ndarray],
    full_df: pd.DataFrame,
    c_name: str,
    record: dict[str, Any],
) -> dict[str, Any]:
    """One (split, draw, class)'s random/dispersed census rows, from exp-10's allocations.

    ``record`` is exp-10's own allocation record for this (split, draw,
    class): ``{"seeds", "focal", "clustered", "random", "dispersed"}``, where
    ``seeds`` is the five-patient random arm and ``dispersed`` is exp-10's
    twenty-patient dispersed allocation.
    """
    seeds = cast(list[str], record["seeds"])
    focal = cast(str, record["focal"])
    dispersed = cast(list[str], record["dispersed"])[:_N_DISPERSED]
    ran20 = cast(list[str], record["random"])

    return {
        "record": {
            "seeds": seeds,
            "focal": focal,
            "random": seeds,
            "dispersed": dispersed,
        },
        **_coverage_r_values(embeddings, full_df, c_name, seeds, dispersed, ran20),
        **_spread_and_site(embeddings, c_name, focal, seeds, dispersed),
        "test_distances": _test_improvement(
            embeddings, full_df, c_name, seeds, dispersed
        ),
    }


def allocation_frames(
    train_df: pd.DataFrame, class_names: list[str], record: dict[str, dict[str, object]]
) -> dict[str, pd.DataFrame]:
    """Build the random and dispersed allocation frames from a census record.

    ``record`` maps class name to ``{"seeds", "focal", "random", "dispersed"}``,
    where ``random`` and ``dispersed`` already hold their five-patient lists,
    as written to ``data/allocations.json`` by the census stage.
    """
    rows: dict[str, list[int]] = {name: [] for name in ALLOCATIONS}
    for c_name in class_names:
        cls_record = record[c_name]
        cls_df = cast(pd.DataFrame, train_df[train_df["cancer_type"] == c_name])
        for allocation in ALLOCATIONS:
            for patient in cast(list[str], cls_record[allocation]):
                rows[allocation].extend(_patch_rows(cls_df, patient, _BROAD_M))
    return {
        name: train_df.loc[idxs].copy().reset_index(drop=True)
        for name, idxs in rows.items()
    }
