"""Per-(split, draw, class) coverage census: draws, coverage distances, site shares."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from breadth import DEPTH_LADDER, N_DRAWS
from breadth.analyze.diagnostics import tissue_source_site
from breadth.sampling import eligible_patients_by_class

from neighbours.allocation import (
    draw_rng,
    neighbour_patients,
    random_patients,
    seed_patients,
)
from neighbours.embedding import cosine_distances

__all__ = ["split_census"]

_MAX_DEPTH = max(DEPTH_LADDER)  # 32: the eligibility floor of E_rc


def _patients_of(df: pd.DataFrame, split_name: str, class_name: str) -> list[str]:
    """Sorted unique patients of one class in one manifest partition."""
    mask = (df["split"] == split_name) & (df["cancer_type"] == class_name)
    return sorted(df.loc[mask, "case_id"].astype(str).unique())


def _embedding_matrix(
    embeddings: dict[tuple[str, str], np.ndarray], patients: list[str], class_name: str
) -> np.ndarray:
    return np.stack([embeddings[(patient, class_name)] for patient in patients])


def _coverage_distance(val_matrix: np.ndarray, pool_matrix: np.ndarray) -> float:
    """Coverage distance (Eq. coverage): mean nearest-training-patient distance."""
    dist = cosine_distances(val_matrix, pool_matrix)
    return float(np.mean(np.min(dist, axis=1)))


def _site_share(patients: list[str], target_sites: set[str | None]) -> float:
    if not patients:
        return float("nan")
    sites = [tissue_source_site(patient) for patient in patients]
    return float(np.mean([site in target_sites for site in sites]))


def _site_shares(
    neighbour_ids: list[str],
    neighbours: list[tuple[str, str]],
    randoms: list[str],
    seeds: list[str],
) -> dict[str, float]:
    selector_site = {
        patient: tissue_source_site(selector) for patient, selector in neighbours
    }
    neighbour_own_site_share = float(
        np.mean(
            [
                tissue_source_site(patient) == selector_site[patient]
                for patient in neighbour_ids
            ]
        )
    )
    seed_sites = {tissue_source_site(seed) for seed in seeds}
    return {
        "neighbour_selector_site": neighbour_own_site_share,
        "neighbour_any_seed_site": _site_share(neighbour_ids, seed_sites),
        "random_any_seed_site": _site_share(randoms, seed_sites),
    }


def _coverage_r_values(
    embeddings: dict[tuple[str, str], np.ndarray],
    full_df: pd.DataFrame,
    c_name: str,
    e_seeds: np.ndarray,
    seeds: list[str],
    low: tuple[str, list[str]],
    randoms: list[str],
) -> dict[str, float]:
    """The deep, low-coverage, and random coverage distances (Eq. coverage)."""
    low_key, low_patients = low
    e_val = _embedding_matrix(
        embeddings, _patients_of(full_df, "validation", c_name), c_name
    )
    return {
        "r_deep": _coverage_distance(e_val, e_seeds),
        f"r_{low_key}": _coverage_distance(
            e_val, _embedding_matrix(embeddings, seeds + low_patients, c_name)
        ),
        "r_random": _coverage_distance(
            e_val, _embedding_matrix(embeddings, seeds + randoms, c_name)
        ),
    }


def _class_test_geometry(
    embeddings: dict[tuple[str, str], np.ndarray],
    full_df: pd.DataFrame,
    c_name: str,
    e_seeds: np.ndarray,
) -> tuple[list[str], np.ndarray]:
    """One class's test patients and their cosine distance to every seed."""
    test_patients = _patients_of(full_df, "test", c_name)
    e_test = _embedding_matrix(embeddings, test_patients, c_name)
    return test_patients, cosine_distances(e_test, e_seeds)


def _test_distance_dict(
    test_patients: list[str], test_seed_dist: np.ndarray
) -> dict[str, float]:
    """Each test patient's distance to its nearest seed (Eq. distance)."""
    test_dist = np.min(test_seed_dist, axis=1)
    return dict(zip(test_patients, (float(d) for d in test_dist)))


def _class_draw(
    train_df: pd.DataFrame,
    full_df: pd.DataFrame,
    embeddings: dict[tuple[str, str], np.ndarray],
    class_names: list[str],
    c_idx: int,
    c_name: str,
    draw: tuple[int, int],
) -> dict[str, Any]:
    """One (split, draw, class)'s seed/neighbour/random draw and its census rows."""
    split, draw_idx = draw
    seeds = seed_patients(train_df, class_names, c_idx, split, draw_idx)
    pool = eligible_patients_by_class(train_df, c_name, _MAX_DEPTH)
    rng = draw_rng(split, draw_idx, c_idx)

    e_seeds = _embedding_matrix(embeddings, seeds, c_name)
    dist_seed_pool = cosine_distances(
        e_seeds, _embedding_matrix(embeddings, pool, c_name)
    )
    neighbours = neighbour_patients(seeds, pool, dist_seed_pool, rng)
    randoms = random_patients(seeds, pool, rng)
    neighbour_ids = [patient for patient, _selector in neighbours]

    r_values = _coverage_r_values(
        embeddings,
        full_df,
        c_name,
        e_seeds,
        seeds,
        ("neighbours", neighbour_ids),
        randoms,
    )
    test_patients, test_seed_dist = _class_test_geometry(
        embeddings, full_df, c_name, e_seeds
    )

    return {
        "record": {"seeds": seeds, "neighbours": neighbours, "random": randoms},
        **r_values,
        "test_distances": _test_distance_dict(test_patients, test_seed_dist),
        "site_shares": _site_shares(neighbour_ids, neighbours, randoms, seeds),
    }


def kappa(r_deep: float, r_neighbours: float, r_random: float) -> float:
    """Coverage leakage (Eq. leakage)."""
    denom = r_deep - r_random
    return (r_deep - r_neighbours) / denom if denom != 0 else float("nan")


def _r_means(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Mean of every ``r_*`` key present in a set of rows sharing the same keys."""
    r_keys = [key for key in rows[0] if key.startswith("r_")]
    return {key: float(np.mean([row[key] for row in rows])) for key in r_keys}


def _r_kappa(r_means: dict[str, float]) -> float:
    """Leakage kappa from ``r_deep``, ``r_random``, and the one other ``r_*`` mean."""
    low_key = next(k for k in r_means if k not in ("r_deep", "r_random"))
    return kappa(r_means["r_deep"], r_means[low_key], r_means["r_random"])


def _class_census(rows: list[dict[str, Any]]) -> dict[str, Any]:
    r_means = _r_means(rows)
    return {
        **r_means,
        "kappa": _r_kappa(r_means),
        "site_shares": {
            key: float(np.nanmean([row["site_shares"][key] for row in rows]))
            for key in rows[0]["site_shares"]
        },
    }


def _split_summary(class_census: dict[str, Any]) -> dict[str, Any]:
    r_means = _r_means(list(class_census.values()))
    return {
        **r_means,
        "kappa": _r_kappa(r_means),
        "classes": class_census,
    }


def split_census(
    full_df: pd.DataFrame,
    train_df: pd.DataFrame,
    embeddings: dict[tuple[str, str], np.ndarray],
    class_names: list[str],
    split: int,
    class_draw: Callable[..., dict[str, Any]] = _class_draw,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """One split's per-class census, allocation records, and split-level kappa."""
    class_rows: dict[str, list[dict[str, Any]]] = {name: [] for name in class_names}
    allocation_records: dict[int, dict[str, dict[str, Any]]] = {
        d: {} for d in range(N_DRAWS)
    }
    for c_idx, c_name in enumerate(class_names):
        for draw in range(N_DRAWS):
            drawn = class_draw(
                train_df, full_df, embeddings, class_names, c_idx, c_name, (split, draw)
            )
            class_rows[c_name].append(drawn)
            allocation_records[draw][c_name] = drawn["record"]

    class_census = {c_name: _class_census(rows) for c_name, rows in class_rows.items()}
    allocations = {
        "allocations": {str(d): allocation_records[d] for d in range(N_DRAWS)},
        "test_distances": {
            c_name: [row["test_distances"] for row in rows]
            for c_name, rows in class_rows.items()
        },
    }
    return _split_summary(class_census), allocations
