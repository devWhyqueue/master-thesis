"""Manifest, arm, and evaluation-partition loading for one (split, draw) shard.

Reuses exp-25/16's patient-draw, class-permutation, and allocation machinery
directly (``prevalence.fit``, ``centre.cohort``), but never calls exp-2's own
``breadth.fit.init_shard`` (it eagerly loads cached *features*, not needed
here) or ``prevalence.fit._arm_rows`` (it also loads features): image-level
training reads ``image_path`` off the manifest rows instead.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
import torch
from decodability.evidence import load_freeze_meta
from imbalance_benchmark.analysis.query import load_test_identity
from imbalance_benchmark.common import ensure_dirs, split_paths

from breadth import exp2_split_paths

from sites import allocation_dir

from prevalence.fit import _shard_context, arm_row_index, class_counts

from mitigation import method_label

__all__ = [
    "ImageShard",
    "load_shard",
    "train_arm",
    "eval_arm",
    "resolve_device",
    "run_dir",
]


class ImageShard(NamedTuple):
    """One (split, draw) shard's train manifest, class names, and eval partitions."""

    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    val_identity: pd.DataFrame
    test_identity: pd.DataFrame
    classes: list[str]
    paths: dict[str, Path]


def load_shard(config: dict[str, Any], split_idx: int) -> ImageShard:
    """Load one split's train/validation/test manifest rows and locked class names."""
    exp2_paths = exp2_split_paths(config, split_idx)
    manifest_path = exp2_paths["data"] / "manifest.csv"
    classes = list(load_freeze_meta(exp2_paths)["class_names"])
    manifest = pd.read_csv(manifest_path)
    partitions = {
        name: manifest.query("split == @name").reset_index(drop=True)
        for name in ("train", "validation", "test")
    }
    return ImageShard(
        partitions["train"],
        partitions["validation"],
        partitions["test"],
        load_test_identity(manifest_path, is_mil=False, split_name="validation"),
        load_test_identity(manifest_path, is_mil=False, split_name="test"),
        classes,
        split_paths(ensure_dirs(config), split_idx),
    )


def train_arm(
    shard: ImageShard, split_idx: int, draw_idx: int, g: int, data_arm: str
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Image paths, integer targets, and realized per-class counts of one arm's training rows."""
    draw = _shard_context(shard.train_df, shard.classes, split_idx, draw_idx, g)
    counts = class_counts(data_arm, draw.perm, draw.available, draw.pool_counts, draw.g)
    rows, y = arm_row_index(draw.train_df, draw.names, draw.patients, counts)
    image_paths = draw.train_df.loc[rows, "image_path"].astype(str).tolist()
    return image_paths, y, np.asarray(counts, dtype=np.int64)


def _targets(df: pd.DataFrame, classes: list[str]) -> np.ndarray:
    """Encode a manifest slice's class labels as integer targets."""
    class_index = {name: index for index, name in enumerate(classes)}
    return np.array([class_index[c] for c in df["cancer_type"]], dtype=np.int64)


def eval_arm(df: pd.DataFrame, classes: list[str]) -> tuple[list[str], np.ndarray]:
    """Image paths and integer targets of a validation/test manifest slice."""
    return df["image_path"].astype(str).tolist(), _targets(df, classes)


def resolve_device() -> torch.device:
    """CUDA when available, else CPU (local/test runs)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_dir(
    shard: ImageShard, arm: str, method: str, param: float | None, draw_idx: int
) -> Path:
    """Result directory for one (arm, method, param, draw) run."""
    label = f"{arm}_{method_label(method, param)}"
    return allocation_dir(shard.paths, label, draw_idx)
