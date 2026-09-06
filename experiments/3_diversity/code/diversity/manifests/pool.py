"""Slot tables, eligible pools, frozen features, and headroom for one allocation."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from imbalance_benchmark.datasets.data import ImbalanceDataset
from imbalance_benchmark.datasets.features.cache import bank_index

__all__ = [
    "SLOT_KEYS",
    "slot_table",
    "eligible_pool",
    "pool_features",
    "headroom_table",
]

SLOT_KEYS = ("cancer_type", "case_id", "slide_id")


def slot_table(source_df: pd.DataFrame) -> pd.DataFrame:
    """Per-(class, case, slide) pinned patch count ``n_cgs`` from a frozen condition."""
    return source_df.groupby(list(SLOT_KEYS)).size().to_frame("n").reset_index()


def eligible_pool(train_df: pd.DataFrame, slots: pd.DataFrame) -> pd.DataFrame:
    """Rows of the exp-2 train split matching one allocation's pinned slots.

    Plan Stage 1, point 3: every training-split row sharing a slot's triple,
    not just the rows the source condition happened to already select.
    """
    pool = train_df.merge(slots[list(SLOT_KEYS)], on=list(SLOT_KEYS), how="inner")
    return cast(pd.DataFrame, pool.reset_index(drop=True))


def pool_features(pool: pd.DataFrame, class_names: list[str]) -> np.ndarray:
    """Frozen Virchow2 features for every pool row, aligned to ``pool``'s row order.

    Builds a throwaway :class:`ImbalanceDataset` over the pool (plan Stage 1,
    point 4) purely to reuse its feature-bank plumbing; the manifest it reads
    is a scratch file, never a persisted exp-3 artifact.
    """
    if "patch_id" not in pool.columns:
        raise RuntimeError(
            "Eligible pool is missing 'patch_id'; narrow/wide selection needs "
            "a stable per-row identity."
        )
    if "feature_index" not in pool.columns or bool(pool["feature_index"].isna().any()):
        raise RuntimeError(
            "Eligible pool is missing a complete 'feature_index' column; "
            "narrow/wide tie-breaking requires it on every row."
        )
    with tempfile.TemporaryDirectory() as scratch:
        scratch_manifest = Path(scratch) / "pool.csv"
        pool.to_csv(scratch_manifest, index=False)
        dataset = ImbalanceDataset(scratch_manifest, class_names=class_names)
        features = bank_index(dataset.rows)
    return features.cpu().numpy()


def headroom_table(pool: pd.DataFrame, slots: pd.DataFrame) -> pd.DataFrame:
    """Per-slot headroom ``h = eligible pool rows / pinned count``."""
    pool_sizes = (
        pool.groupby(list(SLOT_KEYS)).size().to_frame("pool_size").reset_index()
    )
    merged = slots.merge(pool_sizes, on=list(SLOT_KEYS), how="left")
    merged["pool_size"] = merged["pool_size"].fillna(0).astype(int)
    merged["h"] = merged["pool_size"] / merged["n"]
    return merged
