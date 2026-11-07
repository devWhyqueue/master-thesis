"""Stored balanced/imbalanced fits joined to test-patch identity in one canonical class order.

Every stored fit (exp-25/26's r1 and r100, exp-27/28's P100 and S100) evaluated the exp-2 test
split of its patient split in manifest order with labels encoded in that split's own frozen class
order. Rows are mapped to the alphabetical canonical order (``canonical_class_names``) and checked
against the manifest's class names, row count, and the run's own stored balanced accuracy before
any pooling.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, NamedTuple, cast

import numpy as np
import pandas as pd
from imbalance_benchmark.common import ensure_dirs, read_run_record, split_paths

from breadth import exp2_split_paths
from breadth.analyze.canonical import canonical_class_names, canonical_permutation

from sites import allocation_dir

from spectrum import baseline_config

from gap.metrics import confusion, patient_weights

__all__ = [
    "ARM_SOURCES",
    "Fit",
    "SplitIdentity",
    "canonicalize",
    "check_stored_ba",
    "split_identity",
    "load_fit",
    "read_temperature",
]

ARM_SOURCES: dict[str, str] = {
    "r1": "prevalence_outputs",
    "r100": "prevalence_outputs",
    "P100": "p_outputs",
    "S100": "p_outputs",
}
_BA_TOLERANCE = 1e-9
_IDENTITY_COLUMNS = ("case_id", "slide_id", "cancer_type", "patch_id", "image_path")


class SplitIdentity(NamedTuple):
    """One patient split's test rows in prediction order, with canonical labels and weights."""

    names: list[str]
    perm: np.ndarray
    frame: pd.DataFrame
    labels: np.ndarray
    weights: np.ndarray


class Fit(NamedTuple):
    """One stored fit's canonical predictions, probabilities, temperature, and verified BA."""

    preds: np.ndarray
    probs: np.ndarray | None
    temperature: float
    ba: float


def canonicalize(
    labels_local: np.ndarray,
    preds_local: np.ndarray,
    probs_local: np.ndarray | None,
    perm: np.ndarray,
    manifest_names: np.ndarray,
    canonical: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Reorder one record into ``canonical``'s class order; reject any identity mismatch."""
    if len(labels_local) != len(manifest_names):
        raise RuntimeError(
            f"record has {len(labels_local)} rows, manifest test split {len(manifest_names)}"
        )
    to_canonical = np.argsort(perm)
    labels = to_canonical[labels_local]
    if not np.array_equal(np.asarray(canonical)[labels], manifest_names):
        raise RuntimeError("record labels disagree with the manifest's class names")
    probs = None if probs_local is None else probs_local[:, perm]
    return labels, to_canonical[preds_local], probs


def check_stored_ba(
    labels: np.ndarray, preds: np.ndarray, weights: np.ndarray, stored: float
) -> float:
    """Recompute patient-macro BA and require it to equal the run's stored endpoint."""
    n_classes = int(labels.max()) + 1
    ba = float(np.diag(confusion(labels, preds, weights, n_classes)).mean())
    if abs(ba - stored) > _BA_TOLERANCE:
        raise RuntimeError(f"balanced accuracy {ba} differs from stored {stored}")
    return ba


def split_identity(config: dict[str, Any], split: int) -> SplitIdentity:
    """Test rows of one split (``load_test_identity``'s order) with canonical labels."""
    names = canonical_class_names(config)
    manifest = exp2_split_paths(config, split)["data"] / "manifest.csv"
    frame = pd.read_csv(
        manifest, usecols=lambda c: c in {*_IDENTITY_COLUMNS, "split"}, dtype=str
    )
    frame = cast(pd.DataFrame, frame[frame["split"] == "test"]).reset_index(drop=True)
    labels = np.array([names.index(n) for n in frame["cancer_type"]])
    weights = patient_weights(labels, frame["case_id"].to_numpy())
    perm = canonical_permutation(config, split, names)
    return SplitIdentity(names, perm, frame, labels, weights)


def _result_dir(config: dict[str, Any], arm: str, split: int, draw: int) -> Path:
    source = baseline_config(config, ARM_SOURCES[arm])
    return allocation_dir(split_paths(ensure_dirs(source), split), arm, draw)


def load_fit(
    config: dict[str, Any],
    identity: SplitIdentity,
    arm: str,
    split: int,
    draw: int,
    with_probs: bool = True,
) -> Fit:
    """Load, canonicalize, and verify one stored fit against its split's test identity."""
    result_dir = _result_dir(config, arm, split, draw)
    fields = ("labels", "preds", "probabilities") if with_probs else ("labels", "preds")
    record = read_run_record(result_dir, splits=("test",), array_fields=fields)
    if record is None or fields[-1] not in record["splits"].get("test", {}):
        raise RuntimeError(f"missing stored fit {arm} split {split} draw {draw}")
    test = record["splits"]["test"]
    labels, preds, probs = canonicalize(
        np.asarray(test["labels"]),
        np.asarray(test["preds"]),
        np.asarray(test["probabilities"]) if with_probs else None,
        identity.perm,
        identity.frame["cancer_type"].to_numpy(),
        identity.names,
    )
    stored = test["endpoints"]["patient_macro_balanced_accuracy"]
    ba = check_stored_ba(labels, preds, identity.weights, stored)
    return Fit(preds, probs, read_temperature(result_dir, required=with_probs), ba)


def read_temperature(result_dir: Path, required: bool) -> float:
    """Stored validation temperature; NaN when absent unless the caller needs it."""
    path = result_dir / "temperature.json"
    if path.exists():
        return float(json.loads(path.read_text(encoding="utf-8"))["temperature"])
    if required:
        raise RuntimeError(f"missing validation temperature at {path}")
    return float("nan")
