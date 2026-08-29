from __future__ import annotations

import logging
import time
from typing import cast

import numpy as np
import pandas as pd

from imbalance_benchmark.manifest import log_every
from imbalance_benchmark.manifest.sampling.patch import (
    _build_patch_hierarchy,
    _select_from_hierarchy,
)

logger = logging.getLogger(__name__)

# _contribution_cap (patch.py) caps any single patient at 10% and any slide at
# 5% of a class's allocated patches, so any allocation already requires at
# least 10 patients and 20 slides. These floors and those caps are the same
# constraint written twice - do not weaken either without the other.
MIN_POOL_PATIENTS = 10
MIN_POOL_SLIDES = 20


def _pool_patch_count(slide_sizes: pd.Series, slides: list[str]) -> int:
    """Patches covered by a slide selection, without rescanning the frame."""
    return int(slide_sizes.reindex(pd.unique(np.asarray(slides))).sum())


def _pool_is_ready(
    slide_sizes: pd.Series,
    patients: list[str],
    slides: list[str],
    maximum_patches: int | None,
) -> bool:
    """Whether a hierarchy prefix meets fixed-pool diversity and capacity."""
    if len(patients) < MIN_POOL_PATIENTS or len(slides) < MIN_POOL_SLIDES:
        return False
    return maximum_patches is None or (
        _pool_patch_count(slide_sizes, slides) >= maximum_patches
    )


def _copy_hierarchy(
    h: dict[str, dict[str, list[int]]],
) -> dict[str, dict[str, list[int]]]:
    """Copy only the round-robin cursors ``_loop_patches`` consumes, not the whole tree."""
    return {p: {s: list(pids) for s, pids in slides.items()} for p, slides in h.items()}


def _pool_has_capacity(
    df_class: pd.DataFrame,
    patients: list[str],
    slides: list[str],
    required_counts: tuple[int, ...],
    seed: int,
) -> bool:
    """Every count must be selectable; only the largest must retain the whole pool.

    A tail condition's smaller count is drawn *from* the pool, not required to
    exhaust it - only the largest requested count fixes the pool's designated
    patients and slides, since it is the one sized to need all of them.
    """
    if not required_counts:
        return True
    pool = cast(
        pd.DataFrame,
        df_class[
            df_class["case_id"].isin(patients) & df_class["slide_id"].isin(slides)
        ],
    )
    if pool.empty:
        # Matches select_patches_round_robin's own short-circuit: an empty pool
        # trivially satisfies every count (both sides of the subset check are empty).
        return True
    # Every count probes the same (pool, seed) hierarchy; a fresh np.random.default_rng(seed)
    # rebuild per count reproduces identical shuffles, so build once and hand each
    # probe its own copy of the consumable cursors instead of re-deriving them.
    pool_patients, hierarchy = _build_patch_hierarchy(pool, np.random.default_rng(seed))
    maximum = max(required_counts)
    for count in required_counts:
        try:
            selected = _select_from_hierarchy(
                pool, pool_patients, _copy_hierarchy(hierarchy), count
            )
        except ValueError:
            return False
        if count == maximum and (
            not set(pool["case_id"]).issubset(selected["case_id"])
            or not set(pool["slide_id"]).issubset(selected["slide_id"])
        ):
            return False
    return True


def _next_patient_with_remaining(
    patients: list[str], remaining: dict[str, list[str]], cursor: int
) -> tuple[str | None, int]:
    for offset in range(len(patients)):
        index = (cursor + offset) % len(patients)
        patient = patients[index]
        if remaining.get(patient):
            return patient, (index + 1) % len(patients)
    return None, cursor


def _expand_pool(
    pool_hierarchy: tuple[list[str], dict],
    sel_p: list[str],
    sel_s: list[str],
    remaining: dict,
    df: pd.DataFrame,
    required_counts: tuple[int, ...],
    max_pool_units: int | None,
    seed: int,
) -> None:
    """Expand sel_p/sel_s breadth-first until the pool is ready or resources are exhausted."""
    pats, hier = pool_hierarchy
    maximum = max(required_counts, default=None)
    slide_sizes = df["slide_id"].value_counts()
    class_name = df["cancer_type"].iloc[0] if not df.empty else "?"
    patient_index = len(sel_p)
    patient_cursor = 0
    last_logged = time.perf_counter()
    while not _pool_is_ready(
        slide_sizes, sel_p, sel_s, maximum
    ) or not _pool_has_capacity(df, sel_p, sel_s, required_counts, seed):
        last_logged = log_every(
            last_logged,
            logger,
            f"freeze: pool {class_name}: {len(sel_p)} patients, {len(sel_s)} slides, "
            f"{_pool_patch_count(slide_sizes, sel_s)}/{maximum or 0} patches",
        )
        if max_pool_units is not None and len(sel_s) >= max_pool_units:
            break
        patient, patient_cursor = _next_patient_with_remaining(
            sel_p, remaining, patient_cursor
        )
        if patient is not None:
            sel_s.append(remaining[patient].pop(0))
            continue
        if patient_index >= len(pats):
            break
        patient = pats[patient_index]
        patient_index += 1
        sel_p.append(patient)
        slides = list(hier[patient])
        sel_s.append(slides[0])
        remaining[patient] = slides[1:]
    if not _pool_is_ready(slide_sizes, sel_p, sel_s, maximum) or not _pool_has_capacity(
        df, sel_p, sel_s, required_counts, seed
    ):
        raise ValueError(
            "Eligible patches cannot form the required fixed evidence pool"
        )


def _validate_independent_units(min_independent_units: int) -> None:
    if min_independent_units < MIN_POOL_PATIENTS:
        raise ValueError("Patch conditions need at least 10 independent patients")


def _designate_floor(
    df: pd.DataFrame,
    seed: int,
    min_independent_units: int,
) -> tuple[list[str], dict, list[str], list[str], dict]:
    """Validate, shuffle, and take the first ``min_independent_units`` patients as the floor."""
    _validate_independent_units(min_independent_units)
    pats, hier = _build_patch_hierarchy(df, np.random.default_rng(seed))
    if len(pats) < min_independent_units:
        raise ValueError("Eligible patches cannot meet the independent-patient floor")
    sel_p = pats[:min_independent_units]
    sel_s = [next(iter(hier[p])) for p in sel_p]
    remaining = {p: list(hier[p])[1:] for p in sel_p}
    return pats, hier, sel_p, sel_s, remaining


def designate_patch_pool(
    df: pd.DataFrame,
    min_independent_units: int,
    seed: int,
    max_p: int | None = None,
    max_pool_units: int | None = None,
    required_counts: tuple[int, ...] | None = None,
) -> pd.DataFrame:
    """Choose the fixed patient/slide pool used by every patch condition."""
    pats, hier, sel_p, sel_s, remaining = _designate_floor(
        df, seed, min_independent_units
    )
    counts = required_counts or (() if max_p is None else (max_p,))
    _expand_pool(
        (pats, hier),
        sel_p,
        sel_s,
        remaining,
        df,
        counts,
        max_pool_units,
        seed,
    )
    return cast(
        pd.DataFrame, df[df["case_id"].isin(sel_p) & df["slide_id"].isin(sel_s)]
    )
