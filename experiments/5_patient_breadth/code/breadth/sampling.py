"""Patient eligibility, slide round-robin patch sampling, and draw generation."""

from __future__ import annotations

from collections.abc import Callable
import hashlib
from typing import Any, cast

import numpy as np
import pandas as pd

from breadth import (
    BREADTH_LADDER,
    DEPTH_LADDER,
    FALLBACK_BREADTH_LADDER,
    FALLBACK_DEPTH_LADDER,
)

__all__ = [
    "derive_draw_seed",
    "eligible_patients_by_class",
    "check_grid_eligibility",
    "sample_patient_patches_round_robin",
    "sample_cell_draw",
]


def derive_draw_seed(
    base_seed: int,
    split_index: int,
    g: int,
    m: int,
    draw_index: int,
    class_index: int,
) -> int:
    """Deterministically derive a 32-bit seed for one class draw in a cell."""
    token = f"{base_seed}:{split_index}:G{g}:m{m}:draw{draw_index}:cls{class_index}"
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="big") % (2**31 - 1)


def eligible_patients_by_class(
    train_df: pd.DataFrame, class_name: str, m: int
) -> list[str]:
    """Return sorted unique case_ids holding at least m patches of class_name."""
    subset = train_df[train_df["cancer_type"] == class_name]
    if subset.empty:
        return []
    counts: dict[str, int] = {}
    for case in subset["case_id"].astype(str):
        counts[case] = counts.get(case, 0) + 1
    return sorted(case for case, cnt in counts.items() if cnt >= m)


def _audit_counts(
    train_dfs: dict[int, pd.DataFrame],
    class_names: list[str],
    depth_ladder: tuple[int, ...],
) -> dict[str, dict[str, dict[int, int]]]:
    """Record eligible patient counts for each split, class, and depth."""
    return {
        str(s): {
            c: {d: len(eligible_patients_by_class(df, c, d)) for d in depth_ladder}
            for c in class_names
        }
        for s, df in train_dfs.items()
    }


def _resolve_ladders(
    satisfies: Callable[[tuple[int, ...], tuple[int, ...]], bool],
    b_lad: tuple[int, ...],
    d_lad: tuple[int, ...],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Check initial ladders, falling back if infeasible."""
    if satisfies(b_lad, d_lad):
        return b_lad, d_lad
    if satisfies(b_lad, FALLBACK_DEPTH_LADDER):
        return b_lad, FALLBACK_DEPTH_LADDER
    if satisfies(FALLBACK_BREADTH_LADDER, FALLBACK_DEPTH_LADDER):
        return FALLBACK_BREADTH_LADDER, FALLBACK_DEPTH_LADDER
    raise RuntimeError("Even fallback ladders infeasible.")


def check_grid_eligibility(
    train_dfs: dict[int, pd.DataFrame],
    class_names: list[str],
    breadth_ladder: tuple[int, ...] = BREADTH_LADDER,
    depth_ladder: tuple[int, ...] = DEPTH_LADDER,
) -> tuple[tuple[int, ...], tuple[int, ...], dict[str, Any]]:
    """Verify eligible patient counts and resolve realized breadth/depth ladders."""

    def _satisfies(b: tuple[int, ...], d: tuple[int, ...]) -> bool:
        max_g = max(b)
        return all(
            len(eligible_patients_by_class(df, c, depth)) >= max_g
            for df in train_dfs.values()
            for c in class_names
            for depth in d
        )

    audit = _audit_counts(train_dfs, class_names, depth_ladder)
    realized_b, realized_d = _resolve_ladders(_satisfies, breadth_ladder, depth_ladder)
    return realized_b, realized_d, audit


def _slide_patch_lists(patient_df: pd.DataFrame) -> list[list[int]]:
    """Return ordered patch index lists for each slide of a patient."""
    slides = sorted(patient_df["slide_id"].astype(str).unique())
    patch_lists = []
    for s in slides:
        sub = patient_df[patient_df["slide_id"].astype(str) == s]
        if "patch_id" in sub.columns:
            order = np.argsort(np.asarray(sub["patch_id"]), kind="stable")
            patch_lists.append([int(x) for x in np.asarray(sub.index)[order]])
        else:
            patch_lists.append([int(x) for x in np.asarray(sub.index)])
    return patch_lists


def sample_patient_patches_round_robin(patient_df: pd.DataFrame, m: int) -> list[int]:
    """Select m patches round-robin across patient slides to accumulate depth."""
    if len(patient_df) < m:
        raise ValueError(f"Patient has only {len(patient_df)} patches, need {m}")

    slide_patches = _slide_patch_lists(patient_df)
    selected: list[int] = []
    active = [p for p in slide_patches if p]

    while len(selected) < m:
        next_active = []
        for patches in active:
            if patches:
                selected.append(patches.pop(0))
                if len(selected) == m:
                    break
                if patches:
                    next_active.append(patches)
        active = next_active
    return selected


def sample_cell_draw(
    train_df: pd.DataFrame,
    class_names: list[str],
    g: int,
    m: int,
    split_index: int,
    draw_index: int,
    base_seed: int = 0,
) -> pd.DataFrame:
    """Sample one (G, m) draw: G patients per class, m patches per patient."""
    selected_indices: list[int] = []
    for cls_idx, c_name in enumerate(class_names):
        eligible = eligible_patients_by_class(train_df, c_name, m)
        if len(eligible) < g:
            raise RuntimeError(f"Class {c_name} has only {len(eligible)} eligible")
        seed = derive_draw_seed(base_seed, split_index, g, m, draw_index, cls_idx)
        chosen = np.random.default_rng(seed).choice(eligible, size=g, replace=False)
        cls_df = train_df[train_df["cancer_type"] == c_name]
        for case in chosen:
            p_df = cast(
                pd.DataFrame, cls_df[cls_df["case_id"].astype(str) == str(case)]
            )
            selected_indices.extend(sample_patient_patches_round_robin(p_df, m))
    return train_df.loc[selected_indices].copy().reset_index(drop=True)
