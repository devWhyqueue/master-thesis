"""Pool stage: draw-independent pool centres, discriminant basis, and between-patient structure per split."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, NamedTuple, cast

import numpy as np
import pandas as pd
from decodability.evidence import load_freeze_meta
from imbalance_benchmark.common import (
    compute_sha256,
    ensure_dirs,
    split_paths,
    write_json,
)

from breadth import exp2_split_paths
from breadth.sampling import eligible_patients_by_class, load_features_for_df

from centre import POOL_DEPTH
from centre.arms import between_patient_eigenbasis, discriminant_basis
from centre.cohort import patient_rows

__all__ = ["Pool", "pool_path", "train_frame", "run_pool", "load_pool"]

logger = logging.getLogger(__name__)


class Pool(NamedTuple):
    """Pool centres (C, d), discriminant basis, between-patient eigenbasis, and patient deviations."""

    class_names: list[str]
    centres: np.ndarray
    discriminant: np.ndarray
    b_basis: np.ndarray
    b_eigvals: np.ndarray
    deviations: np.ndarray
    deviation_class: np.ndarray


def pool_path(config: dict[str, Any], split_idx: int) -> Path:
    """Location of one split's pool arrays."""
    return split_paths(ensure_dirs(config), split_idx)["data"] / "pool.npz"


def train_frame(
    config: dict[str, Any], split_idx: int
) -> tuple[pd.DataFrame, list[str]]:
    """Training manifest rows and the split's frozen class order."""
    exp2_p = exp2_split_paths(config, split_idx)
    names = list(load_freeze_meta(exp2_p)["class_names"])
    manifest = pd.read_csv(exp2_p["data"] / "manifest.csv")
    return manifest.query("split == 'train'").reset_index(drop=True), names


def _patient_means(train_df: pd.DataFrame, name: str) -> np.ndarray:
    """(n, d) float64 mean over the first POOL_DEPTH round-robin patches of every eligible patient."""
    patients = eligible_patients_by_class(train_df, name, POOL_DEPTH)
    class_df = cast(pd.DataFrame, train_df[train_df["cancer_type"] == name])
    rows = patient_rows(class_df, patients, POOL_DEPTH)
    x = load_features_for_df(train_df.loc[rows]).astype(np.float64)
    return x.reshape(len(patients), POOL_DEPTH, -1).mean(axis=1)


def _build(train_df: pd.DataFrame, names: list[str]) -> Pool:
    centres, deviations, owners = [], [], []
    for ci, name in enumerate(names):
        means = _patient_means(train_df, name)
        logger.info("Class %s: %d eligible patients", name, len(means))
        centres.append(means.mean(axis=0))
        deviations.append(means - centres[-1])
        owners.append(np.full(len(means), ci))
    dev = np.concatenate(deviations)
    b_basis, b_eigvals = between_patient_eigenbasis(dev, len(dev) - len(names))
    centre_arr = np.stack(centres)
    return Pool(
        names,
        centre_arr,
        discriminant_basis(centre_arr),
        b_basis,
        b_eigvals,
        dev,
        np.concatenate(owners),
    )


def run_pool(config: dict[str, Any], split_idx: int) -> Path:
    """Compute and store one split's pool arrays with their checksum."""
    train_df, names = train_frame(config, split_idx)
    pool = _build(train_df, names)
    out = pool_path(config, split_idx)
    arrays = {k: np.asarray(v) for k, v in pool._asdict().items() if k != "class_names"}
    np.savez(out, **cast(dict[str, Any], arrays))
    write_json(
        out.with_suffix(".json"),
        {
            "class_names": names,
            "sha256": compute_sha256(out),
            "patients_per_class": np.bincount(pool.deviation_class).tolist(),
            "between_patient_rank": len(pool.b_eigvals),
        },
    )
    return out


def load_pool(config: dict[str, Any], split_idx: int, class_names: list[str]) -> Pool:
    """Load one split's pool arrays after checking checksum and class order."""
    path = pool_path(config, split_idx)
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if (
        compute_sha256(path) != meta["sha256"]
        or list(meta["class_names"]) != class_names
    ):
        raise RuntimeError(f"Pool arrays at {path} do not match this split")
    with np.load(path) as data:
        return Pool(class_names, **{k: data[k] for k in data.files})
