"""Patient eligibility, slide round-robin patch sampling, and draw generation."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
from imbalance_benchmark.analysis.query import load_test_identity
from imbalance_benchmark.datasets.data import ImbalanceDataset
from imbalance_benchmark.datasets.features import load_feature_row

from breadth import (
    BREADTH_LADDER,
    DEPTH_LADDER,
)

__all__ = [
    "derive_draw_seed",
    "eligible_patients_by_class",
    "check_grid_eligibility",
    "sample_patient_patches_round_robin",
    "sample_class_patches",
    "sample_cell_draw",
    "load_features_for_df",
    "load_eval_partition",
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
    """Record common maximum-depth pool sizes for every cell depth."""
    return {
        str(s): {
            c: {
                d: len(eligible_patients_by_class(df, c, max(depth_ladder)))
                for d in depth_ladder
            }
            for c in class_names
        }
        for s, df in train_dfs.items()
    }


def check_grid_eligibility(
    train_dfs: dict[int, pd.DataFrame],
    class_names: list[str],
    breadth_ladder: tuple[int, ...] = BREADTH_LADDER,
    depth_ladder: tuple[int, ...] = DEPTH_LADDER,
) -> tuple[tuple[int, ...], tuple[int, ...], dict[str, Any]]:
    """Require a common maximum-depth pool for every class and split."""
    audit = _audit_counts(train_dfs, class_names, depth_ladder)
    if any(
        count < max(breadth_ladder)
        for classes in audit.values()
        for depths in classes.values()
        for count in depths.values()
    ):
        raise RuntimeError(
            "Common maximum-depth pool infeasible; reduce the entire grid "
            "in BREADTH_LADDER and DEPTH_LADDER, then rerun preflight."
        )
    return breadth_ladder, depth_ladder, audit


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


def sample_class_patches(
    train_df: pd.DataFrame,
    class_names: list[str],
    class_index: int,
    cell: tuple[int, int],
    draw: tuple[int, int],
    base_seed: int = 0,
) -> list[int]:
    """Sample one class's G patients x m nested-prefix patches for one draw.

    ``cell`` is ``(g, m)``; ``draw`` is ``(split_index, draw_index)``. Patient
    selection always keys off ``max(DEPTH_LADDER)`` eligibility, so the same
    patients are paired across every depth of one breadth.
    """
    g, m = cell
    split_index, draw_index = draw
    c_name = class_names[class_index]
    eligible = eligible_patients_by_class(train_df, c_name, max(DEPTH_LADDER))
    if len(eligible) < g:
        raise RuntimeError(f"Class {c_name} has only {len(eligible)} eligible")
    seed = derive_draw_seed(
        base_seed, split_index, g, max(DEPTH_LADDER), draw_index, class_index
    )
    chosen = np.random.default_rng(seed).choice(eligible, size=g, replace=False)
    cls_df = train_df[train_df["cancer_type"] == c_name]
    selected: list[int] = []
    for case in chosen:
        p_df = cast(pd.DataFrame, cls_df[cls_df["case_id"].astype(str) == str(case)])
        selected.extend(sample_patient_patches_round_robin(p_df, m))
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
    """Pair patients across depths; reveal nested round-robin patch prefixes."""
    if m not in DEPTH_LADDER:
        raise ValueError("Depth must belong to the configured grid")
    selected_indices: list[int] = []
    for cls_idx in range(len(class_names)):
        selected_indices.extend(
            sample_class_patches(
                train_df,
                class_names,
                cls_idx,
                (g, m),
                (split_index, draw_index),
                base_seed,
            )
        )
    return train_df.loc[selected_indices].copy().reset_index(drop=True)


def load_features_for_df(df: pd.DataFrame) -> np.ndarray:
    """Load features for a slice of manifest rows directly."""
    paths = df["feature_path"].astype(str).to_numpy()
    f_idx = df["feature_index"].to_numpy() if "feature_index" in df.columns else None
    rows = [
        load_feature_row(
            paths[i],
            int(f_idx[i]) if f_idx is not None and pd.notna(f_idx[i]) else None,
        )
        for i in range(len(df))
    ]
    return torch.stack(rows).numpy()


def load_eval_partition(
    manifest_path: str | Path, class_names: list[str], split_name: str
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load features, integer targets, and patient identity for validation/test."""
    ds = ImbalanceDataset(manifest_path, split_name=split_name, class_names=class_names)
    return (
        load_features_for_df(ds.df),
        ds.get_int_targets(),
        load_test_identity(manifest_path, is_mil=False, split_name=split_name),
    )
