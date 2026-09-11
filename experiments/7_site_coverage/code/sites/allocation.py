"""Site pools and the three nested breadth-depth-site training allocations."""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from breadth import DEPTH_LADDER
from breadth.analyze.diagnostics import tissue_source_site
from breadth.sampling import (
    derive_draw_seed,
    eligible_patients_by_class,
    sample_class_patches,
    sample_patient_patches_round_robin,
)

from sites import (
    ALLOCATIONS,
    BACKGROUND_CELL,
    N_CORE_SITES,
    N_SITES,
    PATIENTS_PER_SITE,
    SITE_BASE_SEED,
)

__all__ = ["site_pool", "site_class_names", "draw_allocations"]

SiteRecord = dict[str, dict[str, list[str]]]


def site_pool(train_df: pd.DataFrame, class_name: str) -> dict[str, list[str]]:
    """Eligible patients of one class grouped by site (Eq. site-pool).

    Only sites with at least ``PATIENTS_PER_SITE`` eligible patients enter
    the pool.
    """
    eligible = eligible_patients_by_class(train_df, class_name, max(DEPTH_LADDER))
    by_site: dict[str, list[str]] = {}
    for patient in eligible:
        site = tissue_source_site(patient)
        if site is not None:
            by_site.setdefault(site, []).append(patient)
    return {s: p for s, p in by_site.items() if len(p) >= PATIENTS_PER_SITE}


def site_class_names(
    train_dfs: dict[int, pd.DataFrame], class_names: list[str]
) -> list[str]:
    """Classes whose site pool has >= N_SITES sites in every split."""
    return [
        name
        for name in class_names
        if all(len(site_pool(df, name)) >= N_SITES for df in train_dfs.values())
    ]


def _draw_sites(
    pool: dict[str, list[str]], split: int, draw: int, class_index: int
) -> tuple[np.random.Generator, list[str], set[str], list[str]]:
    """Draw 10 sites, split into 5 core + 5 added, from one class's site pool."""
    seed = derive_draw_seed(
        SITE_BASE_SEED, split, N_SITES, PATIENTS_PER_SITE, draw, class_index
    )
    rng = np.random.default_rng(seed)
    sites_sorted = sorted(pool)
    chosen = [str(s) for s in rng.choice(sites_sorted, size=N_SITES, replace=False)]
    core = {str(s) for s in rng.choice(chosen, size=N_CORE_SITES, replace=False)}
    added = [s for s in chosen if s not in core]
    return rng, chosen, core, added


def _allocate_background(
    train_df: pd.DataFrame,
    class_names: list[str],
    c_idx: int,
    split: int,
    draw: int,
    rows: dict[str, list[int]],
) -> None:
    """Draw one fixed background class's patches, shared by all three allocations."""
    background = sample_class_patches(
        train_df, class_names, c_idx, BACKGROUND_CELL, (split, draw)
    )
    for name in ALLOCATIONS:
        rows[name].extend(background)


def _patch_rows(cls_df: pd.DataFrame, patient: str, m: int) -> list[int]:
    """Round-robin patch indices of one patient's slides."""
    p_df = cast(pd.DataFrame, cls_df[cls_df["case_id"].astype(str) == patient])
    return sample_patient_patches_round_robin(p_df, m)


def _allocate_site(
    cls_df: pd.DataFrame,
    pool: dict[str, list[str]],
    site: str,
    is_core: bool,
    rng: np.random.Generator,
    rows: dict[str, list[int]],
) -> None:
    """Draw one site's ordered patients and add their patches to each allocation."""
    patients = [
        str(p)
        for p in rng.choice(sorted(pool[site]), size=PATIENTS_PER_SITE, replace=False)
    ]
    if is_core:
        rows["deep"].extend(_patch_rows(cls_df, patients[0], 32))
        for patient in patients:
            rows["broad5"].extend(_patch_rows(cls_df, patient, 8))
    for patient in patients[:2]:
        rows["broad10"].extend(_patch_rows(cls_df, patient, 8))


def _allocate_site_class(
    train_df: pd.DataFrame,
    c_idx: int,
    c_name: str,
    split: int,
    draw: int,
    rows: dict[str, list[int]],
) -> dict[str, list[str]]:
    """Draw one site class's ten sites and their patients' nested patches."""
    pool = site_pool(train_df, c_name)
    rng, chosen_sites, core_sites, added_sites = _draw_sites(pool, split, draw, c_idx)
    cls_df = cast(pd.DataFrame, train_df[train_df["cancer_type"] == c_name])
    for site in chosen_sites:
        _allocate_site(cls_df, pool, site, site in core_sites, rng, rows)
    return {"core": sorted(core_sites), "added": sorted(added_sites)}


def draw_allocations(
    train_df: pd.DataFrame,
    class_names: list[str],
    site_classes: list[str],
    split: int,
    draw: int,
) -> tuple[dict[str, pd.DataFrame], SiteRecord]:
    """Sample the three nested site allocations, plus fixed background classes."""
    rows: dict[str, list[int]] = {name: [] for name in ALLOCATIONS}
    site_record: SiteRecord = {}
    for c_idx, c_name in enumerate(class_names):
        if c_name in site_classes:
            site_record[c_name] = _allocate_site_class(
                train_df, c_idx, c_name, split, draw, rows
            )
        else:
            _allocate_background(train_df, class_names, c_idx, split, draw, rows)
    frames = {
        name: train_df.loc[idxs].copy().reset_index(drop=True)
        for name, idxs in rows.items()
    }
    return frames, site_record
