"""Seed, neighbour, and random patient selection; patch materialisation."""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from breadth.sampling import (
    derive_draw_seed,
    sample_class_patches,
    sample_patient_patches_round_robin,
)

from neighbours import ALLOCATIONS, DEEP_CELL, NEIGHBOUR_BASE_SEED, N_ROUNDS

__all__ = [
    "seed_patients",
    "draw_rng",
    "neighbour_patients",
    "random_patients",
    "allocation_frames",
]

_BROAD_G, _BROAD_M = 20, 8
_N_ADDED = _BROAD_G - DEEP_CELL[0]  # 15


def seed_patients(
    train_df: pd.DataFrame, class_names: list[str], c_idx: int, split: int, draw: int
) -> list[str]:
    """Ordered unique seed patients: the exp-5 five-patient, 32-patch draw."""
    patches = sample_class_patches(
        train_df, class_names, c_idx, DEEP_CELL, (split, draw)
    )
    cases = train_df.loc[patches, "case_id"].astype(str).tolist()
    ordered: list[str] = []
    seen: set[str] = set()
    for case in cases:
        if case not in seen:
            seen.add(case)
            ordered.append(case)
    return ordered


def draw_rng(split: int, draw: int, c_idx: int) -> np.random.Generator:
    """Random generator for one (split, draw, class)'s neighbour/random draw."""
    seed = derive_draw_seed(NEIGHBOUR_BASE_SEED, split, _BROAD_G, _BROAD_M, draw, c_idx)
    return np.random.default_rng(seed)


def neighbour_patients(
    seeds: list[str], pool: list[str], dist: np.ndarray, rng: np.random.Generator
) -> list[tuple[str, str]]:
    """Greedy round-robin nearest-neighbour selection (Eq. nesting).

    Seeds take turns, in a random order, picking their nearest not-yet-selected
    pool patient over ``N_ROUNDS`` rounds; ties break on patient identifier.
    Returns ``(patient, selector_seed)`` pairs in selection order.
    """
    pool_arr = np.asarray(pool)
    seed_set = set(seeds)
    available = np.array([patient not in seed_set for patient in pool], dtype=bool)
    order = rng.permutation(len(seeds))
    selected: list[tuple[str, str]] = []
    for _ in range(N_ROUNDS):
        for seed_pos in order:
            seed = seeds[int(seed_pos)]
            candidates = np.flatnonzero(available)
            distances = dist[seed_pos, candidates]
            candidate_ids = pool_arr[candidates]
            best = candidates[np.lexsort((candidate_ids, distances))[0]]
            available[best] = False
            selected.append((str(pool_arr[best]), seed))
    return selected


def random_patients(
    seeds: list[str], pool: list[str], rng: np.random.Generator
) -> list[str]:
    """Draw the fifteen added patients without replacement from pool minus seeds."""
    candidates = sorted(set(pool) - set(seeds))
    chosen = rng.choice(np.asarray(candidates), size=_N_ADDED, replace=False)
    return [str(patient) for patient in chosen]


def _patch_rows(cls_df: pd.DataFrame, patient: str, m: int) -> list[int]:
    """Round-robin patch indices of one patient's slides."""
    p_df = cast(pd.DataFrame, cls_df[cls_df["case_id"].astype(str) == patient])
    return sample_patient_patches_round_robin(p_df, m)


def allocation_frames(
    train_df: pd.DataFrame, class_names: list[str], record: dict[str, dict[str, object]]
) -> dict[str, pd.DataFrame]:
    """Build the neighbours and random allocation frames from a census patient record.

    ``record`` maps class name to ``{"seeds", "neighbours", "random"}``, as
    written to ``data/allocations.json`` by the census stage.
    """
    rows: dict[str, list[int]] = {name: [] for name in ALLOCATIONS}
    for c_name in class_names:
        cls_record = record[c_name]
        cls_df = cast(pd.DataFrame, train_df[train_df["cancer_type"] == c_name])
        seeds = cast(list[str], cls_record["seeds"])
        for allocation in ALLOCATIONS:
            for patient in seeds:
                rows[allocation].extend(_patch_rows(cls_df, patient, _BROAD_M))
        neighbours = cast(list[list[str]], cls_record["neighbours"])
        for patient, _selector in neighbours:
            rows["neighbours"].extend(_patch_rows(cls_df, patient, _BROAD_M))
        for patient in cast(list[str], cls_record["random"]):
            rows["random"].extend(_patch_rows(cls_df, patient, _BROAD_M))
    return {
        name: train_df.loc[idxs].copy().reset_index(drop=True)
        for name, idxs in rows.items()
    }
