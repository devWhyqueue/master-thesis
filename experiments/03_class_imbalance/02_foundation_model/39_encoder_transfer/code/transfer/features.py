"""Requested-row resolution for exp-39's phase-03 feature scope.

Extraction scope, per ``configs/input_audit.json``: the union of every scheduled
training row (draws 10-19, all 7 arms, all 3 splits) plus every frozen
validation/test row (all 3 splits). ``transfer.extract`` caches this union by
``slide_id``, so the same physical image extracted for one split's train arm
and another split's val/test row shares one feature. ``transfer.manifest``
reuses these same helpers to audit Virchow2's existing pointers and join both
encoders' model-specific manifests.
"""

from __future__ import annotations

from typing import Any, cast

import pandas as pd

from imbalance_benchmark.common import N_PATIENT_SPLITS

from breadth import exp2_split_paths

from prevalence import patients_per_class

from transfer import MAIN_DRAWS
from transfer.schedule import draw_schedule, load_train_identity

__all__ = [
    "IDENTITY_COLS",
    "ordered_identity",
    "requested_mask",
    "requested_frame",
]

IDENTITY_COLS = ("case_id", "slide_id", "patch_id")


def ordered_identity(group: pd.DataFrame) -> list[str]:
    """Stable per-slide patch order: sort by patch ID, hash identity plus source path."""
    ordered = group.sort_values("patch_id")
    return [
        f"{case}\0{slide}\0{patch}\0{path}"
        for case, slide, patch, path in zip(
            ordered["case_id"].astype(str),
            ordered["slide_id"].astype(str),
            ordered["patch_id"].astype(str),
            ordered["image_path"].astype(str),
            strict=True,
        )
    ]


def _full_manifest(config: dict[str, Any], split_idx: int) -> pd.DataFrame:
    m_file = exp2_split_paths(config, split_idx)["data"] / "manifest.csv"
    return pd.read_csv(m_file)


def _train_union_identity(
    train_df: pd.DataFrame, names: list[str], split_idx: int, g: int
) -> set[tuple[str, str, str]]:
    identity: set[tuple[str, str, str]] = set()
    for draw_idx in MAIN_DRAWS:
        cell = draw_schedule(train_df, names, split_idx, draw_idx, g)
        for arm in cell["arms"].values():
            for row in arm["patch_identity"]:
                case, slide, patch = row
                identity.add((str(case), str(slide), str(patch)))
    return identity


def requested_mask(
    full: pd.DataFrame, wanted_train_identity: set[tuple[str, str, str]]
) -> pd.Series:
    """True for every val/test row and for scheduled training rows; False otherwise."""
    is_train = full["split"] == "train"
    keys = list(
        zip(
            full["case_id"].astype(str),
            full["slide_id"].astype(str),
            full["patch_id"].astype(str),
        )
    )
    return pd.Series(
        [(not t) or (k in wanted_train_identity) for t, k in zip(is_train, keys)],
        index=full.index,
    )


def _split_requested_rows(config: dict[str, Any], split_idx: int) -> pd.DataFrame:
    full = _full_manifest(config, split_idx)
    train_df, names = load_train_identity(config, split_idx)
    wanted = _train_union_identity(
        train_df, names, split_idx, patients_per_class(config)
    )
    return cast(pd.DataFrame, full[requested_mask(full, wanted)])


def requested_frame(config: dict[str, Any]) -> pd.DataFrame:
    """Union of every requested row across all 3 splits, deduplicated by patch identity."""
    parts = [
        _split_requested_rows(config, split_idx)
        for split_idx in range(N_PATIENT_SPLITS)
    ]
    combined = pd.concat(parts, ignore_index=True)
    cols = list(IDENTITY_COLS)
    mismatched = cast(pd.Series, combined.groupby(cols)["image_path"].nunique())
    duplicated = cast(pd.Series, mismatched[mismatched > 1])
    if not duplicated.empty:
        bad = duplicated.index.tolist()[:5]
        raise ValueError(
            f"Same patch identity resolves to different image paths: {bad}"
        )
    return cast(
        pd.DataFrame,
        combined.drop_duplicates(subset=cols).reset_index(drop=True)[
            cols + ["image_path"]
        ],
    )
