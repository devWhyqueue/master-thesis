"""Focal-seed, concentrated-neighbour, and replayed-random patient selection."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd

from breadth.analyze.diagnostics import tissue_source_site
from breadth.sampling import derive_draw_seed, eligible_patients_by_class

from neighbours.allocation import _patch_rows, draw_rng, random_patients, seed_patients
from neighbours.coverage import (
    _MAX_DEPTH,
    _class_test_geometry,
    _coverage_r_values,
    _embedding_matrix,
    _site_share,
    _test_distance_dict,
)
from neighbours.embedding import cosine_distances

from concentrated import ALLOCATIONS, FOCAL_BASE_SEED

__all__ = [
    "focal_seed",
    "concentrated_patients",
    "exp8_random_patients",
    "class_draw",
    "allocation_frames",
]

_BROAD_G, _BROAD_M = 20, 8
_N_ADDED = 15


def focal_seed(seeds: list[str], split: int, draw: int, c_idx: int) -> int:
    """Uniformly drawn focal-seed position, on a stream separate from the others.

    Deriving from ``FOCAL_BASE_SEED`` rather than ``NEIGHBOUR_BASE_SEED`` keeps
    this draw off the random arm's ``draw_rng`` stream, so
    :func:`exp8_random_patients` still replays exp-8's exact random patients.
    """
    seed = derive_draw_seed(FOCAL_BASE_SEED, split, _BROAD_G, _BROAD_M, draw, c_idx)
    return int(np.random.default_rng(seed).integers(len(seeds)))


def concentrated_patients(
    focal_pos: int, seeds: list[str], pool: list[str], dist_seed_pool: np.ndarray
) -> list[str]:
    """The fifteen non-seed pool patients nearest the focal seed (Eq. distance).

    Ties break toward the smaller patient identifier, the same idiom as
    exp-8's ``neighbour_patients``.
    """
    pool_arr = np.asarray(pool)
    seed_set = set(seeds)
    available = np.array([patient not in seed_set for patient in pool], dtype=bool)
    candidates = np.flatnonzero(available)
    distances = dist_seed_pool[focal_pos, candidates]
    candidate_ids = pool_arr[candidates]
    order = np.lexsort((candidate_ids, distances))
    chosen = candidates[order[:_N_ADDED]]
    return [str(patient) for patient in pool_arr[chosen]]


def exp8_random_patients(
    seeds: list[str], pool: list[str], split: int, draw: int, c_idx: int
) -> list[str]:
    """Exp-8's random allocation, replayed bit-for-bit on its own draw_rng.

    Exp-8 drew its random patients only after ``neighbour_patients`` burned
    one ``rng.permutation(len(seeds))`` (the greedy round-robin selection
    order) off the same generator. Replaying that burn here reproduces
    exp-8's exact random draws, as the report promises, without deriving a
    new seed.
    """
    rng = draw_rng(split, draw, c_idx)
    rng.permutation(len(seeds))
    return random_patients(seeds, pool, rng)


def _concentrated_record(
    seeds: list[str],
    focal: str,
    concentrated: list[str],
    randoms: list[str],
    focal_cell: list[str],
) -> dict[str, Any]:
    """Assemble one class's seed/focal/concentrated/random allocation record."""
    return {
        "seeds": seeds,
        "focal": focal,
        "concentrated": concentrated,
        "random": randoms,
        "focal_cell": focal_cell,
    }


def _focal_site_shares(
    focal: str, concentrated: list[str], randoms: list[str]
) -> dict[str, float]:
    """Share of the concentrated and random additions from the focal seed's site."""
    focal_site = {tissue_source_site(focal)}
    return {
        "concentrated_focal_site": _site_share(concentrated, focal_site),
        "random_focal_site": _site_share(randoms, focal_site),
    }


def _concentrated_geometry(
    embeddings: dict[tuple[str, str], np.ndarray],
    full_df: pd.DataFrame,
    c_name: str,
    e_seeds: np.ndarray,
    seeds: list[str],
    concentrated: list[str],
    randoms: list[str],
) -> tuple[dict[str, float], list[str], np.ndarray]:
    """Coverage distances and test-to-seed geometry for one (split, draw, class)."""
    r_values = _coverage_r_values(
        embeddings,
        full_df,
        c_name,
        e_seeds,
        seeds,
        ("concentrated", concentrated),
        randoms,
    )
    test_patients, test_seed_dist = _class_test_geometry(
        embeddings, full_df, c_name, e_seeds
    )
    return r_values, test_patients, test_seed_dist


def _focal_cell_patients(
    test_patients: list[str], test_seed_dist: np.ndarray, focal_pos: int
) -> list[str]:
    """Test patients of one class whose nearest seed is the focal seed."""
    nearest_seed = np.argmin(test_seed_dist, axis=1)
    return [p for p, n in zip(test_patients, nearest_seed) if n == focal_pos]


def class_draw(
    train_df: pd.DataFrame,
    full_df: pd.DataFrame,
    embeddings: dict[tuple[str, str], np.ndarray],
    class_names: list[str],
    c_idx: int,
    c_name: str,
    draw: tuple[int, int],
) -> dict[str, Any]:
    """One (split, draw, class)'s seed/focal/concentrated/random draw and census rows."""
    split, draw_idx = draw
    seeds = seed_patients(train_df, class_names, c_idx, split, draw_idx)
    pool = eligible_patients_by_class(train_df, c_name, _MAX_DEPTH)
    focal_pos = focal_seed(seeds, split, draw_idx, c_idx)

    e_seeds = _embedding_matrix(embeddings, seeds, c_name)
    dist_seed_pool = cosine_distances(
        e_seeds, _embedding_matrix(embeddings, pool, c_name)
    )
    concentrated = concentrated_patients(focal_pos, seeds, pool, dist_seed_pool)
    randoms = exp8_random_patients(seeds, pool, split, draw_idx, c_idx)

    r_values, test_patients, test_seed_dist = _concentrated_geometry(
        embeddings, full_df, c_name, e_seeds, seeds, concentrated, randoms
    )
    focal_cell = _focal_cell_patients(test_patients, test_seed_dist, focal_pos)
    focal = seeds[focal_pos]

    return {
        "record": _concentrated_record(seeds, focal, concentrated, randoms, focal_cell),
        **r_values,
        "test_distances": _test_distance_dict(test_patients, test_seed_dist),
        "site_shares": _focal_site_shares(focal, concentrated, randoms),
    }


def allocation_frames(
    train_df: pd.DataFrame, class_names: list[str], record: dict[str, dict[str, object]]
) -> dict[str, pd.DataFrame]:
    """Build the concentrated and random allocation frames from a census record.

    ``record`` maps class name to ``{"seeds", "concentrated", "random", ...}``,
    as written to ``data/allocations.json`` by the census stage.
    """
    rows: dict[str, list[int]] = {name: [] for name in ALLOCATIONS}
    for c_name in class_names:
        cls_record = record[c_name]
        cls_df = cast(pd.DataFrame, train_df[train_df["cancer_type"] == c_name])
        seeds = cast(list[str], cls_record["seeds"])
        for allocation in ALLOCATIONS:
            for patient in seeds:
                rows[allocation].extend(_patch_rows(cls_df, patient, _BROAD_M))
        for patient in cast(list[str], cls_record["concentrated"]):
            rows["concentrated"].extend(_patch_rows(cls_df, patient, _BROAD_M))
        for patient in cast(list[str], cls_record["random"]):
            rows["random"].extend(_patch_rows(cls_df, patient, _BROAD_M))
    return {
        name: train_df.loc[idxs].copy().reset_index(drop=True)
        for name, idxs in rows.items()
    }
