"""Training-only class centres, the separation index J, and the alpha-scaled centre-shift
intervention x^(alpha) = x + (alpha - 1)(mu_y - mu_bar) (PLAN.md "Controlled experiment").

Centres are computed once per split, from every eligible training patient at exp-25/26's
DEPTH = 160 pool, mirroring ``centre.pool``'s patient-balanced mean-of-patient-means but without
its discriminant/whitening machinery, which this experiment does not use.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, NamedTuple, cast

import numpy as np
import pandas as pd
from imbalance_benchmark.common import (
    compute_sha256,
    ensure_dirs,
    split_paths,
    write_json,
)

from breadth.sampling import eligible_patients_by_class, load_features_for_df

from centre.cohort import patient_rows

from separation import CENTRE_DEPTH

__all__ = [
    "Centres",
    "class_centres",
    "separation_index",
    "apply_intervention",
    "centres_path",
    "run_centres",
    "load_centres",
]


class Centres(NamedTuple):
    """Training-only, patient-balanced class centres, their grand mean, and the pooled
    within-class RMS of patient means (all at ``CENTRE_DEPTH``)."""

    class_names: list[str]
    means: np.ndarray  # (C, d)
    grand_mean: np.ndarray  # (d,)
    within_rms: float


def _patient_means(
    train_df: pd.DataFrame, name: str, depth: int = CENTRE_DEPTH
) -> np.ndarray:
    """(n, d) mean over the first ``depth`` round-robin patches of every eligible patient."""
    patients = eligible_patients_by_class(train_df, name, depth)
    class_df = cast(pd.DataFrame, train_df[train_df["cancer_type"] == name])
    rows = patient_rows(class_df, patients, depth)
    x = load_features_for_df(train_df.loc[rows]).astype(np.float64)
    return x.reshape(len(patients), depth, -1).mean(axis=1)


def class_centres(
    train_df: pd.DataFrame, names: list[str], depth: int = CENTRE_DEPTH
) -> Centres:
    """Training-only, patient-balanced class centres and pooled within-class RMS, at ``depth``."""
    means, deviations = [], []
    for name in names:
        patient_means = _patient_means(train_df, name, depth)
        centre = patient_means.mean(axis=0)
        means.append(centre)
        deviations.append(patient_means - centre)
    means_arr = np.stack(means)
    dev = np.concatenate(deviations)
    within_rms = float(np.sqrt((dev**2).sum(axis=1).mean()))
    return Centres(names, means_arr, means_arr.mean(axis=0), within_rms)


def separation_index(centres: Centres) -> float:
    """J: median nearest-class-centre distance, divided by pooled within-class RMS."""
    d = np.linalg.norm(centres.means[:, None, :] - centres.means[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    return float(np.median(d.min(axis=1)) / centres.within_rms)


def apply_intervention(
    x: np.ndarray, y: np.ndarray, centres: Centres, alpha: float
) -> np.ndarray:
    """x^(alpha) = x + (alpha - 1)(mu_y - mu_bar); alpha = 1 is the identity."""
    if alpha == 1.0:
        return x
    shift = (alpha - 1.0) * (centres.means - centres.grand_mean)
    return x + shift[y]


def centres_path(config: dict[str, Any], split_idx: int) -> Path:
    """Location of one split's signed centre arrays."""
    return split_paths(ensure_dirs(config), split_idx)["data"] / "centres.npz"


def run_centres(
    config: dict[str, Any], split_idx: int, train_df: pd.DataFrame, names: list[str]
) -> Path:
    """Compute and store one split's centres, grand mean, within-RMS, and J."""
    centres = class_centres(train_df, names)
    j = separation_index(centres)
    out = centres_path(config, split_idx)
    np.savez(out, means=centres.means, grand_mean=centres.grand_mean)
    write_json(
        out.with_suffix(".json"),
        {
            "class_names": names,
            "within_rms": centres.within_rms,
            "separation_index": j,
            "sha256": compute_sha256(out),
        },
    )
    return out


def load_centres(
    config: dict[str, Any], split_idx: int, class_names: list[str]
) -> Centres:
    """Load one split's signed centre arrays after checking checksum and class order."""
    path = centres_path(config, split_idx)
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if (
        compute_sha256(path) != meta["sha256"]
        or list(meta["class_names"]) != class_names
    ):
        raise RuntimeError(f"Centre arrays at {path} do not match this split")
    with np.load(path) as data:
        return Centres(
            class_names, data["means"], data["grand_mean"], meta["within_rms"]
        )
