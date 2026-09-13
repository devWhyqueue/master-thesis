"""Clustered, dispersed, and random patient selection; patch materialisation."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd

from breadth.analyze.diagnostics import tissue_source_site
from breadth.sampling import eligible_patients_by_class

from neighbours.allocation import _patch_rows, seed_patients
from neighbours.coverage import (
    _MAX_DEPTH,
    _coverage_distance,
    _embedding_matrix,
    _patients_of,
    _site_share,
)
from neighbours.embedding import cosine_distances

from concentrated.allocation import exp8_random_patients, focal_seed

from composition import ALLOCATIONS

__all__ = [
    "clustered_patients",
    "dispersed_patients",
    "class_draw",
    "allocation_frames",
]

_N_ADDED = 19
_BROAD_M = 8


def clustered_patients(focal: str, pool: list[str], dist_pool: np.ndarray) -> list[str]:
    """Focal patient plus its nineteen nearest pool neighbours (Eq. distance).

    Ties break toward the smaller patient identifier.
    """
    pool_arr = np.asarray(pool)
    f_idx = pool.index(focal)
    candidates = np.array([i for i in range(len(pool)) if i != f_idx])
    distances = dist_pool[f_idx, candidates]
    order = np.lexsort((pool_arr[candidates], distances))
    chosen = candidates[order[:_N_ADDED]]
    return [focal] + [str(patient) for patient in pool_arr[chosen]]


def dispersed_patients(focal: str, pool: list[str], dist_pool: np.ndarray) -> list[str]:
    """Focal patient plus nineteen facility-location patients (Eq. facility).

    Each step adds the pool patient that most shortens the mean distance from
    every pool patient to its nearest selected patient; ties break toward the
    smaller patient identifier.
    """
    pool_arr = np.asarray(pool)
    f_idx = pool.index(focal)
    selected = {f_idx}
    best = dist_pool[:, f_idx].copy()
    chosen = [focal]
    for _ in range(_N_ADDED):
        objective = np.minimum(best[:, None], dist_pool).mean(axis=0)
        objective[list(selected)] = np.inf
        j = int(np.lexsort((pool_arr, objective))[0])
        selected.add(j)
        best = np.minimum(best, dist_pool[:, j])
        chosen.append(str(pool_arr[j]))
    return chosen


def _coverage_r_values(
    embeddings: dict[tuple[str, str], np.ndarray],
    full_df: pd.DataFrame,
    c_name: str,
    seeds: list[str],
    clustered: list[str],
    random_alloc: list[str],
    dispersed: list[str],
) -> dict[str, float]:
    """Coverage distances of the five seeds and the three allocations (Eq. coverage)."""
    e_val = _embedding_matrix(
        embeddings, _patients_of(full_df, "validation", c_name), c_name
    )
    return {
        "r_5": _coverage_distance(e_val, _embedding_matrix(embeddings, seeds, c_name)),
        "r_clustered": _coverage_distance(
            e_val, _embedding_matrix(embeddings, clustered, c_name)
        ),
        "r_random": _coverage_distance(
            e_val, _embedding_matrix(embeddings, random_alloc, c_name)
        ),
        "r_dispersed": _coverage_distance(
            e_val, _embedding_matrix(embeddings, dispersed, c_name)
        ),
    }


def _spread(
    embeddings: dict[tuple[str, str], np.ndarray], c_name: str, patients: list[str]
) -> float:
    """Mean off-diagonal pairwise cosine distance within a cohort."""
    e = _embedding_matrix(embeddings, patients, c_name)
    dist = cosine_distances(e, e)
    n = len(patients)
    return float((dist.sum() - np.trace(dist)) / (n * (n - 1)))


def _site_shares(
    focal: str, clustered: list[str], random_alloc: list[str], dispersed: list[str]
) -> dict[str, float]:
    """Share of each allocation's non-focal additions from the focal patient's site."""
    focal_site = {tissue_source_site(focal)}
    return {
        "site_clustered": _site_share([p for p in clustered if p != focal], focal_site),
        "site_random": _site_share([p for p in random_alloc if p != focal], focal_site),
        "site_dispersed": _site_share([p for p in dispersed if p != focal], focal_site),
    }


def _test_improvement(
    embeddings: dict[tuple[str, str], np.ndarray],
    full_df: pd.DataFrame,
    c_name: str,
    clustered: list[str],
    random_alloc: list[str],
) -> dict[str, float]:
    """Each test patient's coverage improvement: distance to nearest clustered
    training patient minus distance to nearest random training patient."""
    test_patients = _patients_of(full_df, "test", c_name)
    e_test = _embedding_matrix(embeddings, test_patients, c_name)
    d_clustered = np.min(
        cosine_distances(e_test, _embedding_matrix(embeddings, clustered, c_name)),
        axis=1,
    )
    d_random = np.min(
        cosine_distances(e_test, _embedding_matrix(embeddings, random_alloc, c_name)),
        axis=1,
    )
    improvement = d_clustered - d_random
    return dict(zip(test_patients, (float(x) for x in improvement)))


def _draw_allocations(
    train_df: pd.DataFrame,
    class_names: list[str],
    c_idx: int,
    c_name: str,
    split: int,
    draw_idx: int,
    embeddings: dict[tuple[str, str], np.ndarray],
) -> tuple[list[str], str, list[str], list[str], list[str]]:
    """Seeds, focal patient, and the clustered/random/dispersed allocations."""
    seeds = seed_patients(train_df, class_names, c_idx, split, draw_idx)
    pool = eligible_patients_by_class(train_df, c_name, _MAX_DEPTH)
    focal = seeds[focal_seed(seeds, split, draw_idx, c_idx)]

    e_pool = _embedding_matrix(embeddings, pool, c_name)
    dist_pool = cosine_distances(e_pool, e_pool)
    clustered = clustered_patients(focal, pool, dist_pool)
    dispersed = dispersed_patients(focal, pool, dist_pool)
    random_alloc = seeds + exp8_random_patients(seeds, pool, split, draw_idx, c_idx)
    return seeds, focal, clustered, random_alloc, dispersed


def _draw_census(
    embeddings: dict[tuple[str, str], np.ndarray],
    full_df: pd.DataFrame,
    c_name: str,
    seeds: list[str],
    focal: str,
    clustered: list[str],
    random_alloc: list[str],
    dispersed: list[str],
) -> dict[str, Any]:
    """Coverage, spread, site-share, and test-improvement rows for one draw."""
    return {
        **_coverage_r_values(
            embeddings, full_df, c_name, seeds, clustered, random_alloc, dispersed
        ),
        "spread_clustered": _spread(embeddings, c_name, clustered),
        "spread_random": _spread(embeddings, c_name, random_alloc),
        "spread_dispersed": _spread(embeddings, c_name, dispersed),
        **_site_shares(focal, clustered, random_alloc, dispersed),
        "test_distances": _test_improvement(
            embeddings, full_df, c_name, clustered, random_alloc
        ),
    }


def class_draw(
    train_df: pd.DataFrame,
    full_df: pd.DataFrame,
    embeddings: dict[tuple[str, str], np.ndarray],
    class_names: list[str],
    c_idx: int,
    c_name: str,
    draw: tuple[int, int],
) -> dict[str, Any]:
    """One (split, draw, class)'s clustered/random/dispersed draw and census rows."""
    split, draw_idx = draw
    seeds, focal, clustered, random_alloc, dispersed = _draw_allocations(
        train_df, class_names, c_idx, c_name, split, draw_idx, embeddings
    )
    return {
        "record": {
            "seeds": seeds,
            "focal": focal,
            "clustered": clustered,
            "random": random_alloc,
            "dispersed": dispersed,
        },
        **_draw_census(
            embeddings,
            full_df,
            c_name,
            seeds,
            focal,
            clustered,
            random_alloc,
            dispersed,
        ),
    }


def allocation_frames(
    train_df: pd.DataFrame, class_names: list[str], record: dict[str, dict[str, object]]
) -> dict[str, pd.DataFrame]:
    """Build the clustered, random, and dispersed allocation frames from a census record.

    ``record`` maps class name to ``{"seeds", "focal", "clustered", "random",
    "dispersed"}``, where each allocation key already holds its full
    twenty-patient list, as written to ``data/allocations.json`` by the
    census stage.
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
