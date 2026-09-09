"""Load frozen features, labels, and patient identities for one cell."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
from imbalance_benchmark.analysis.query import load_test_identity
from imbalance_benchmark.common import verify_signed_file
from imbalance_benchmark.datasets.data import ImbalanceDataset, load_training_dataset
from imbalance_benchmark.datasets.features.cache import bank_index
from imbalance_benchmark.manifest.freeze import verify_manifest_freeze

from decodability import (
    INPUT_DIM,
    allocation_manifest,
    exp2_split_paths,
)

__all__ = ["CellEvidence", "load_cell", "load_freeze_meta"]


@dataclass(frozen=True)
class CellEvidence:
    """Frozen features, labels, and identities for one dataset-split-support cell."""

    support: str
    class_names: tuple[str, ...]
    train_x: torch.Tensor
    train_y: np.ndarray
    train_patches: list[str]
    train_patients: list[str]
    val_x: torch.Tensor
    val_y: np.ndarray
    val_identity: pd.DataFrame
    test_x: torch.Tensor
    test_y: np.ndarray
    test_identity: pd.DataFrame


def load_freeze_meta(exp2_paths: dict[str, Path]) -> dict[str, Any]:
    """Verify exp-2 freeze signature and return its metadata dict."""
    freeze_path = exp2_paths["data"] / "manifest_freeze.json"
    if not freeze_path.exists():
        raise FileNotFoundError(f"Missing exp-2 freeze: {freeze_path}")
    verify_signed_file(freeze_path)
    meta = json.loads(freeze_path.read_text(encoding="utf-8"))
    verify_manifest_freeze(meta)
    return meta


def _extract_split(
    manifest_path: Path, is_mil: bool, split: str, class_names: list[str]
) -> tuple[torch.Tensor, np.ndarray, pd.DataFrame]:
    """Load features, integer labels, and identity for validation or test split."""
    ds = cast(
        ImbalanceDataset,
        load_training_dataset(manifest_path, is_mil, split, class_names=class_names),
    )
    return (
        bank_index(ds.rows),
        ds.get_int_targets(),
        load_test_identity(manifest_path, is_mil, split_name=split),
    )


def _load_train_evidence(
    manifest_path: Path, is_mil: bool, class_names: list[str]
) -> tuple[torch.Tensor, np.ndarray, list[str], list[str]]:
    """Load training features, labels, patch IDs, and patient IDs."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing allocation manifest: {manifest_path}")
    ds = cast(
        ImbalanceDataset,
        load_training_dataset(
            manifest_path, is_mil, split_name=None, class_names=class_names
        ),
    )
    df = pd.read_csv(manifest_path)
    return (
        bank_index(ds.rows),
        ds.get_int_targets(),
        [str(x) for x in df["patch_id"]],
        [str(x) for x in df["case_id"]],
    )


def _load_eval_splits(
    eval_m: Path, is_mil: bool, classes: list[str]
) -> tuple[
    tuple[torch.Tensor, np.ndarray, pd.DataFrame],
    tuple[torch.Tensor, np.ndarray, pd.DataFrame],
]:
    """Load validation and test splits from manifest."""
    return _extract_split(eval_m, is_mil, "validation", classes), _extract_split(
        eval_m, is_mil, "test", classes
    )


def load_cell(config: dict[str, Any], split_index: int, support: str) -> CellEvidence:
    """Load verified train/val/test data for one split and support condition."""
    exp2_p = exp2_split_paths(config, split_index)
    freeze = load_freeze_meta(exp2_p)
    classes = list(freeze["class_names"])
    is_mil = freeze.get("runtime_config", {}).get("dataset", {}).get("regime") == "wsi"

    tr_x, tr_y, tr_pts, tr_cases = _load_train_evidence(
        allocation_manifest(exp2_p, support), is_mil, classes
    )
    val_data, test_data = _load_eval_splits(
        exp2_p["data"] / "manifest.csv", is_mil, classes
    )
    if tr_x.shape[1] != INPUT_DIM:
        raise ValueError(f"Feature dim mismatch: {tr_x.shape[1]} != {INPUT_DIM}")

    return CellEvidence(
        support, tuple(classes), tr_x, tr_y, tr_pts, tr_cases, *val_data, *test_data
    )
