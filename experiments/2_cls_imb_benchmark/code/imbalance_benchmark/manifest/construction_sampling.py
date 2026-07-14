from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from imbalance_benchmark.common import compute_data_hash


def _build_patch_hierarchy(
    df_class: pd.DataFrame, rng: np.random.Generator
) -> tuple[list[str], dict[str, dict[str, list[int]]]]:
    """Build nested randomized dictionary for patch sampling."""
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
    """Round-robin patches breadth-first: one patch per patient per pass.

    Visiting every patient once (then every slide within a patient) before any
    unit is revisited makes each smaller allocation a nested prefix of the
    larger ones and maximizes patient/slide diversity: the fixed per-class
    patient and slide pool is preserved across balanced and imbalanced
    conditions rather than concentrating a small allocation into a few units.
    """
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
    """Return an exact contribution cap, rejecting allocations below one unit."""
    cap = int(np.floor(n_examples * fraction))
    if cap < 1:
        raise ValueError(
            f"{n_examples} examples cannot satisfy the {fraction:.0%} {unit} cap"
        )
    return cap


def designate_patch_pool(
    df: pd.DataFrame,
    min_p: int,
    seed: int,
    max_p: int | None = None,
) -> pd.DataFrame:
    """Choose the fixed patient/slide pool used by every patch condition."""
    if min_p < 20:
        raise ValueError("Patch conditions need at least 20 patches per class")
    rng = np.random.default_rng(seed)
    pats, hier = _build_patch_hierarchy(df, rng)
    sel_p, sel_s = [], []
    for p in pats:
        sel_p.append(p)
        sel_s.extend(hier[p])
        if _pool_is_ready(df, sel_p, sel_s, max_p):
            break
    else:
        raise ValueError(
            "Eligible patches cannot form the required fixed evidence pool"
        )
    return cast(
        pd.DataFrame, df[df["case_id"].isin(sel_p) & df["slide_id"].isin(sel_s)]
    )


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


def select_patches_round_robin(
    df_class: pd.DataFrame, n_patches: int, seed: int
) -> pd.DataFrame:
    """Sample patches with round-robin patient and slide caps (10% patient, 5% slide)."""
    if df_class.empty or n_patches <= 0:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    patients, h = _build_patch_hierarchy(df_class, rng)
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


def _loop_slides(patients: list[str], h: dict, max_p: int, n: int) -> list[int]:
    """Execute loop for round robin slide sampling."""
    selected, pat_counts = [], {p: 0 for p in patients}
    prog = True
    while len(selected) < n and prog:
        prog = False
        for p in patients:
            if len(selected) >= n or pat_counts[p] >= max_p:
                continue
            if h[p]:
                selected.append(h[p].pop(0))
                pat_counts[p] += 1
                prog = True
    return selected


def _build_slide_hierarchy(
    df_slides: pd.DataFrame, rng: np.random.Generator
) -> tuple[list[str], dict[str, list[int]]]:
    """Build a randomized per-patient slide-index dictionary."""
    patients = list(cast(np.ndarray, df_slides["case_id"].unique()))
    rng.shuffle(patients)
    h = {
        p: list(df_slides[df_slides["case_id"] == p].index.to_numpy()) for p in patients
    }
    for p in h:
        rng.shuffle(h[p])
    return patients, h


def select_slides_round_robin(
    df_class: pd.DataFrame, n_slides: int, seed: int
) -> pd.DataFrame:
    """Sample slides for MIL with round-robin patient caps (10%)."""
    if df_class.empty or n_slides <= 0:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    df_slides = df_class.drop_duplicates("slide_id")
    patients, h = _build_slide_hierarchy(df_slides, rng)
    selected = _loop_slides(
        patients, h, _contribution_cap(n_slides, 0.10, "patient"), n_slides
    )
    if len(selected) < n_slides:
        raise ValueError("Slide allocation is infeasible under the 10% patient cap")
    return cast(
        pd.DataFrame,
        df_class[df_class["slide_id"].isin(df_slides.loc[selected, "slide_id"])],
    )


def build_manifest_hash(manifest_df: pd.DataFrame) -> str:
    """Create a hash of key manifest columns for immutability verification."""
    columns = cast(
        pd.DataFrame, manifest_df[["case_id", "slide_id", "cancer_type", "split"]]
    )
    records = columns.sort_values(by=["split", "cancer_type", "slide_id"]).to_dict(
        "records"
    )
    return compute_data_hash(records)
