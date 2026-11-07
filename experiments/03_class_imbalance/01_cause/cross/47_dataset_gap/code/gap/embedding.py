"""Per-split geometry stage: sample, load frozen Virchow2 rows, neighbour agreement, PCA, examples.

Manifests keep historical feature directories (``2_cls_imb_benchmark``, ``2_benchmark_patch``);
on Hydra these resolve through the compatibility symlinks to ``01_benchmark/02_benchmark_patch``,
so paths are read exactly as stored and only row indices and values are validated.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, NamedTuple, cast

import numpy as np
import pandas as pd
from imbalance_benchmark.common import ensure_dirs, write_json
from imbalance_benchmark.datasets.features.cache import load_slide_features
from sklearn.decomposition import PCA

from breadth import exp2_split_paths
from breadth.analyze.canonical import canonical_class_names

from gap.geometry import (
    assert_disjoint,
    knn_agreement,
    nearest,
    sample_rows,
    summarize_agreement,
    variance_shares,
)
from gap.metrics import patient_weights

__all__ = ["take_rows", "load_features", "run_geometry", "geometry_path"]

logger = logging.getLogger(__name__)

_COLUMNS = (
    "case_id",
    "slide_id",
    "cancer_type",
    "patch_id",
    "image_path",
    "split",
    "feature_path",
    "feature_index",
)
_METRICS = ("cosine", "euclidean")
_N_COMPONENTS = 10


def take_rows(tensor: np.ndarray, indices: np.ndarray, path: str) -> np.ndarray:
    """Rows ``indices`` of one slide tensor; out-of-range, negative, or non-finite rows fail."""
    if indices.min() < 0 or indices.max() >= len(tensor):
        raise RuntimeError(f"feature index outside [0, {len(tensor)}) in {path}")
    rows = np.asarray(tensor[indices], dtype=np.float64)
    if not np.isfinite(rows).all():
        raise RuntimeError(f"non-finite feature values in {path}")
    return rows


def load_features(frame: pd.DataFrame) -> np.ndarray:
    """(n, d) float64 features for manifest rows, each slide tensor read once."""
    out: np.ndarray | None = None
    for path, group in frame.groupby("feature_path", sort=False):
        tensor = load_slide_features(str(path)).numpy()
        rows = take_rows(tensor, group["feature_index"].to_numpy(int), str(path))
        out = np.empty((len(frame), rows.shape[1])) if out is None else out
        out[frame.index.get_indexer(group.index)] = rows
    load_slide_features.cache_clear()
    return cast(np.ndarray, out)


def geometry_path(config: dict[str, Any], split: int) -> Path:
    """Where one split's geometry summary is written."""
    return ensure_dirs(config)["data"] / f"geometry_split{split}.json"


def _samples(config: dict[str, Any], split: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Training reference sample and held-out query sample of one split, reset-indexed."""
    params = config["analysis"]
    manifest = exp2_split_paths(config, split)["data"] / "manifest.csv"
    frame = pd.read_csv(manifest, usecols=lambda c: c in _COLUMNS)
    frame["case_id"] = frame["case_id"].astype(str)
    out = []
    for part, full, offset in (("train", True, 0), ("test", False, 1000)):
        sub = cast(pd.DataFrame, frame[frame["split"] == part])
        rows = sample_rows(
            sub,
            params["n_patients"],
            params["n_patches"],
            params["seed"] + offset + split,
            full_groups=full,
        )
        out.append(sub.loc[rows].reset_index(drop=True))
    assert_disjoint(out[0]["case_id"].to_numpy(), out[1]["case_id"].to_numpy())
    return out[0], out[1]


class Sample(NamedTuple):
    """One split's reference and query rows, their features, canonical labels, query weights."""

    ref: pd.DataFrame
    query: pd.DataFrame
    x_ref: np.ndarray
    x_query: np.ndarray
    ref_y: np.ndarray
    query_y: np.ndarray
    weights: np.ndarray


def _load_sample(config: dict[str, Any], split: int, names: list[str]) -> Sample:
    """Sample one split, load its features, and encode labels in canonical order."""
    ref, query = _samples(config, split)
    query_y = np.array([names.index(n) for n in query["cancer_type"]])
    return Sample(
        ref,
        query,
        load_features(ref),
        load_features(query),
        np.array([names.index(n) for n in ref["cancer_type"]]),
        query_y,
        patient_weights(query_y, query["case_id"].to_numpy()),
    )


def _examples(
    config: dict[str, Any], names: list[str], sample: Sample, agreement: np.ndarray
) -> list[dict[str, Any]]:
    """Per class: the query patch nearest the class's median agreement and its neighbours."""
    ref, query, x_ref, x_query, ref_y = sample[:5]
    partners = {a: b for a, b in config["analysis"]["pairs"]}
    partners |= {b: a for a, b in config["analysis"]["pairs"]}
    rows = []
    for c, name in enumerate(names):
        members = np.flatnonzero(query["cancer_type"].to_numpy() == name)
        a = agreement[members]
        pick = int(members[np.argmin(np.abs(a - np.median(a)))])
        base = {"class": name, "query_patch_id": query.at[pick, "patch_id"]}
        rows.append({**base, "role": "query", **_patch(query, pick, agreement[pick])})
        masks = {"same_nn": ref_y == c, "other_nn": ref_y != c}
        if name in partners:
            masks["partner_nn"] = ref_y == names.index(partners[name])
        for role, mask in masks.items():
            idx, dist = nearest(x_ref[mask], x_query[pick : pick + 1], "cosine")
            j = int(np.flatnonzero(mask)[idx[0]])
            rows.append(
                {
                    **base,
                    "role": role,
                    "cosine_distance": float(dist[0]),
                    **_patch(ref, j, np.nan),
                }
            )
    return rows


def _patch(frame: pd.DataFrame, i: int, agreement: float) -> dict[str, Any]:
    keys = ("patch_id", "case_id", "slide_id", "cancer_type", "image_path")
    return {**{k: str(frame.at[i, k]) for k in keys}, "agreement": float(agreement)}


def _agreement(sample: Sample, ks: tuple[int, ...], metric: str) -> np.ndarray:
    """Per-query neighbour agreement against the reference set for one distance."""
    return knn_agreement(
        sample.x_ref, sample.ref_y, sample.x_query, sample.query_y, ks, metric
    )


def _pca_summary(sample: Sample, split: int) -> dict[str, Any]:
    """Training-fitted PCA's explained variance; split 0 also keeps the projected query points."""
    pca = PCA(n_components=_N_COMPONENTS, random_state=0).fit(sample.x_ref)
    out: dict[str, Any] = {
        "pca_explained_variance": pca.explained_variance_ratio_.tolist()
    }
    if split == 0:
        z = pca.transform(sample.x_query)[:, :2]
        out["pca_points"] = {
            "pc1": z[:, 0].tolist(),
            "pc2": z[:, 1].tolist(),
            "label": sample.query_y.tolist(),
            "weight": sample.weights.tolist(),
        }
    return out


def _sample_summary(sample: Sample, split: int, names: list[str]) -> dict[str, Any]:
    """Sample counts, within-patient variance shares, and PCA of one split."""
    return {
        "split": split,
        "class_names": names,
        "feature_dim": int(sample.x_ref.shape[1]),
        "reference": _counts(sample.ref),
        "query": _counts(sample.query),
        "within_patient_share": variance_shares(
            sample.x_ref, sample.ref["case_id"].to_numpy(), sample.ref_y, len(names)
        ).tolist(),
        **_pca_summary(sample, split),
    }


def run_geometry(config: dict[str, Any], split: int) -> Path:
    """Write one split's agreement, variance shares, PCA, and (split 0) example selection."""
    params, names = config["analysis"], canonical_class_names(config)
    ks = tuple(params["k_values"])
    sample = _load_sample(config, split, names)
    result = _sample_summary(sample, split, names)
    agreements = {m: _agreement(sample, ks, m) for m in _METRICS}
    for metric, values in agreements.items():
        result[metric] = summarize_agreement(
            values, sample.query_y, sample.weights, ks, len(names)
        )
    if split == 0:
        primary = agreements["cosine"][:, ks.index(params["primary_k"])]
        examples = pd.DataFrame(_examples(config, names, sample, primary))
        examples.to_csv(geometry_path(config, 0).with_name("examples.csv"), index=False)
    path = geometry_path(config, split)
    write_json(path, result)
    logger.info("geometry split %d written to %s", split, path)
    return path


def _counts(frame: pd.DataFrame) -> dict[str, Any]:
    """Patients and patches per class actually sampled."""
    grouped = frame.groupby("cancer_type")
    return {
        "patients": grouped["case_id"].nunique().to_dict(),
        "patches": grouped.size().to_dict(),
    }
