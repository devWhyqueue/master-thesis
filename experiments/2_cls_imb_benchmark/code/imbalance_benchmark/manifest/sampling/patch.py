from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd


def _build_patch_hierarchy(
    df_class: pd.DataFrame, rng: np.random.Generator
) -> tuple[list[str], dict[str, dict[str, list[int]]]]:
    patients = cast(np.ndarray, df_class["case_id"].unique())
    rng.shuffle(patients)
    h: dict[str, dict[str, list[int]]] = {}
    for pat in patients:
        h[pat] = {}
        pat_df = cast(pd.DataFrame, df_class[df_class["case_id"] == pat])
        slides = cast(np.ndarray, pat_df["slide_id"].unique())
        rng.shuffle(slides)
        for sld in slides:
            pids = cast(np.ndarray, pat_df[pat_df["slide_id"] == sld].index.to_numpy())
            rng.shuffle(pids)
            h[pat][sld] = list(pids)
    return list(patients), h


def _loop_patches(
    patients: list[str], h: dict, max_p: int, max_s: int, n: int
) -> tuple[list[int], dict, dict]:
    selected, pat_counts, sld_counts = [], {p: 0 for p in patients}, {}
    slide_cursor = {p: 0 for p in patients}
    prog = True
    while len(selected) < n and prog:
        prog = False
        for p in patients:
            if len(selected) >= n or pat_counts[p] >= max_p:
                continue
            slides = list(h[p])
            for offset in range(len(slides)):
                s = slides[(slide_cursor[p] + offset) % len(slides)]
                if sld_counts.get(s, 0) >= max_s or not h[p][s]:
                    continue
                selected.append(h[p][s].pop(0))
                pat_counts[p] += 1
                sld_counts[s] = sld_counts.get(s, 0) + 1
                slide_cursor[p] = (slides.index(s) + 1) % len(slides)
                prog = True
                break
    return selected, pat_counts, sld_counts


def _contribution_cap(n_examples: int, fraction: float, unit: str) -> int:
    cap = int(np.floor(n_examples * fraction))
    if cap < 1:
        raise ValueError(
            f"{n_examples} examples cannot satisfy the {fraction:.0%} {unit} cap"
        )
    return cap


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
    patient_index = len(sel_p)
    patient_cursor = 0
    while not _pool_is_ready(df, sel_p, sel_s, maximum) or not _pool_has_capacity(
        df, sel_p, sel_s, required_counts, seed
    ):
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
    if not _pool_is_ready(df, sel_p, sel_s, maximum) or not _pool_has_capacity(
        df, sel_p, sel_s, required_counts, seed
    ):
        raise ValueError(
            "Eligible patches cannot form the required fixed evidence pool"
        )


def _next_patient_with_remaining(
    patients: list[str], remaining: dict[str, list[str]], cursor: int
) -> tuple[str | None, int]:
    for offset in range(len(patients)):
        index = (cursor + offset) % len(patients)
        patient = patients[index]
        if remaining.get(patient):
            return patient, (index + 1) % len(patients)
    return None, cursor


def designate_patch_pool(
    df: pd.DataFrame,
    min_independent_units: int,
    seed: int,
    max_p: int | None = None,
    max_pool_units: int | None = None,
    required_counts: tuple[int, ...] | None = None,
) -> pd.DataFrame:
    """Choose the fixed patient/slide pool used by every patch condition."""
    if min_independent_units < 10:
        raise ValueError("Patch conditions need at least 10 independent patients")
    rng = np.random.default_rng(seed)
    pats, hier = _build_patch_hierarchy(df, rng)
    if len(pats) < min_independent_units:
        raise ValueError("Eligible patches cannot meet the independent-patient floor")
    sel_p = pats[:min_independent_units]
    sel_s = [next(iter(hier[p])) for p in sel_p]
    remaining = {p: list(hier[p])[1:] for p in sel_p}
    counts = required_counts or (() if max_p is None else (max_p,))
    _expand_pool(
        (pats, hier), sel_p, sel_s, remaining, df, counts, max_pool_units, seed
    )
    return cast(
        pd.DataFrame, df[df["case_id"].isin(sel_p) & df["slide_id"].isin(sel_s)]
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


def _pool_is_ready(
    df_class: pd.DataFrame,
    patients: list[str],
    slides: list[str],
    maximum_patches: int | None,
) -> bool:
    """Whether a hierarchy prefix meets fixed-pool diversity and capacity."""
    selected = df_class[df_class["slide_id"].isin(slides)]
    return (
        len(patients) >= 10
        and len(slides) >= 20
        and (maximum_patches is None or len(selected) >= maximum_patches)
    )


def _select_from_hierarchy(
    df_class: pd.DataFrame,
    patients: list[str],
    h: dict[str, dict[str, list[int]]],
    n_patches: int,
) -> pd.DataFrame:
    """Consume a hierarchy's round-robin cursors to pick ``n_patches``."""
    selected, _, _ = _loop_patches(
        patients,
        h,
        _contribution_cap(n_patches, 0.10, "patient"),
        _contribution_cap(n_patches, 0.05, "slide"),
        n_patches,
    )
    if len(selected) < n_patches:
        raise ValueError(
            "Patch allocation is infeasible under the 10% patient and 5% slide caps"
        )
    return df_class.loc[selected]


def select_patches_round_robin(
    df_class: pd.DataFrame, n_patches: int, seed: int
) -> pd.DataFrame:
    """Sample patches with round-robin patient and slide caps (10% patient, 5% slide)."""
    if df_class.empty or n_patches <= 0:
        return pd.DataFrame()
    patients, h = _build_patch_hierarchy(df_class, np.random.default_rng(seed))
    return _select_from_hierarchy(df_class, patients, h, n_patches)
