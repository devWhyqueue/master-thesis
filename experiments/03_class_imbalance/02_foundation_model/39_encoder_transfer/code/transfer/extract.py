"""Sharded, resumable UNI2-h extraction and cache audit (exp-39, phase 03).

Each shard owns a disjoint slice of the requested-row union (``transfer.features``)
and writes only pending per-slide records; ``merge_features`` is the sole merge
stage, so concurrently running shards never race on the shared cache manifest.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pandas as pd
import torch

from imbalance_benchmark.common import output_root, sign_file, write_json
from imbalance_benchmark.datasets.features.cache import load_slide_features
from imbalance_benchmark.datasets.features.cache_manifest import (
    cached_slide_ids,
    merge_pending_slides,
    record_pending_slide,
    save_tensor_atomic,
    validate_cached_slide,
)

from transfer import uni
from transfer.features import ordered_identity, pilot_requested_frame, requested_frame

__all__ = [
    "uni2h_feature_root",
    "shard_slides",
    "extract_shard",
    "merge_features",
    "audit_uni2h",
    "extract_pilot",
    "pilot_feature_root",
]

EmbedFn = Callable[[list[str], str, torch.device, dict[str, Any]], torch.Tensor]


def uni2h_feature_root(config: dict[str, Any]) -> Path:
    """Shared, per-dataset UNI2-h cache namespace, disjoint from Virchow2's."""
    return output_root(config) / "data" / "features" / "uni2h"


def pilot_feature_root(config: dict[str, Any]) -> Path:
    """Keep reserved-draw tensors separate from the audited main cache."""
    return output_root(config) / "data" / "features" / "uni2h_pilot"


def shard_slides(slide_ids: list[str], shard_index: int, shards: int) -> list[str]:
    """Deterministic disjoint partition of sorted slide IDs across ``shards`` workers."""
    if not 0 <= shard_index < shards:
        raise ValueError("shard_index must be in [0, shards)")
    return sorted(slide_ids)[shard_index::shards]


def _group_by_slide(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        str(slide_id): group
        for slide_id, group in frame.groupby("slide_id", sort=False)
    }


def _extract_one(
    feature_root: Path,
    slide_id: str,
    group: pd.DataFrame,
    dtype: str,
    device: torch.device,
    embed_fn: EmbedFn,
    model_cache: dict[str, Any],
) -> None:
    """Embed and atomically write one slide's pending tensor and cache record."""
    identities = ordered_identity(group)
    image_paths = group.sort_values("patch_id")["image_path"].astype(str).tolist()
    tensor = embed_fn(image_paths, dtype, device, model_cache)
    path = feature_root / f"{slide_id}.pt"
    save_tensor_atomic(tensor, path)
    record_pending_slide(feature_root, slide_id, path, identities, len(tensor))


def _extract_missing(
    feature_root: Path,
    groups: dict[str, pd.DataFrame],
    my_slides: list[str],
    cached: set[str],
    dtype: str,
    device: torch.device,
    embed_fn: EmbedFn,
) -> None:
    model_cache: dict[str, Any] = {}
    for slide_id in my_slides:
        if slide_id not in cached:
            _extract_one(
                feature_root,
                slide_id,
                groups[slide_id],
                dtype,
                device,
                embed_fn,
                model_cache,
            )


def extract_shard(
    config: dict[str, Any],
    shard_index: int,
    shards: int,
    dtype: str = "float32",
    device: torch.device | None = None,
    embed_fn: EmbedFn = uni.extract_slide_features,
    cache: tuple[pd.DataFrame, Path] | None = None,
) -> None:
    """Extract this shard's assigned, not-yet-merged slides; write pending records only.

    Never merges: merging is one separate stage (``merge_features``), so
    concurrently running shards cannot race on the shared cache manifest.
    """
    frame, feature_root = cache or (requested_frame(config), uni2h_feature_root(config))
    feature_root.mkdir(parents=True, exist_ok=True)
    uni.write_or_verify_provenance(feature_root, dtype)
    groups = _group_by_slide(frame)
    my_slides = shard_slides(list(groups), shard_index, shards)
    cached = cached_slide_ids(feature_root)
    resolved_device = device or torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    _extract_missing(
        feature_root, groups, my_slides, cached, dtype, resolved_device, embed_fn
    )


def merge_features(
    config: dict[str, Any],
    frame: pd.DataFrame | None = None,
    feature_root: Path | None = None,
) -> None:
    """One-time merge of every shard's completed pending records into the cache manifest."""
    feature_root = uni2h_feature_root(config) if feature_root is None else feature_root
    frame = requested_frame(config) if frame is None else frame
    expected = {
        slide_id: (feature_root / f"{slide_id}.pt", ordered_identity(group))
        for slide_id, group in _group_by_slide(frame).items()
    }
    merge_pending_slides(feature_root, expected)


def _index_slide(
    feature_root: Path, slide_id: str, group: pd.DataFrame
) -> list[tuple[tuple[str, str, str], tuple[str, int]]] | None:
    """Validate one cached slide; return its identity index entries, or ``None`` if corrupt."""
    ordered = group.sort_values("patch_id")
    identities = ordered_identity(ordered)
    path = feature_root / f"{slide_id}.pt"
    try:
        validate_cached_slide(feature_root, slide_id, path, identities, len(identities))
        tensor = load_slide_features(str(path))
    except (ValueError, OSError):
        return None
    if (
        tensor.shape != (len(identities), uni.FEATURE_DIM)
        or not torch.isfinite(tensor).all()
    ):
        return None
    keys = zip(
        ordered["case_id"].astype(str),
        ordered["slide_id"].astype(str),
        ordered["patch_id"].astype(str),
    )
    return [(key, (str(path), row_index)) for row_index, key in enumerate(keys)]


def audit_uni2h(
    config: dict[str, Any],
    frame: pd.DataFrame | None = None,
    feature_root: Path | None = None,
) -> tuple[dict[str, Any], dict[tuple[str, str, str], tuple[str, int]]]:
    """Verify every requested UNI2-h slide's cache entry; return counts and an identity index."""
    groups = _group_by_slide(requested_frame(config) if frame is None else frame)
    feature_root = uni2h_feature_root(config) if feature_root is None else feature_root
    cached = cached_slide_ids(feature_root)
    missing = sorted(set(groups) - cached)
    corrupt: list[str] = []
    index: dict[tuple[str, str, str], tuple[str, int]] = {}
    for slide_id in sorted(set(groups) & cached):
        entries = _index_slide(feature_root, slide_id, groups[slide_id])
        load_slide_features.cache_clear()
        if entries is None:
            corrupt.append(slide_id)
        else:
            index.update(entries)
    audit = {
        "encoder": "uni2h",
        "requested_slides": len(groups),
        "requested_patches": sum(len(group) for group in groups.values()),
        "matched_patches": len(index),
        "missing_slides": missing,
        "corrupt_slides": corrupt,
        "unresolved_missing": len(missing),
        "unresolved_corrupt": len(corrupt),
    }
    return audit, index


def extract_pilot(config: dict[str, Any]) -> Path:
    """Extract and audit reserved-draw training features without changing main tensors."""
    frame = pilot_requested_frame(config)
    root = pilot_feature_root(config)
    extract_shard(config, 0, 1, cache=(frame, root))
    merge_features(config, frame=frame, feature_root=root)
    audit, _ = audit_uni2h(config, frame=frame, feature_root=root)
    if audit["unresolved_missing"] or audit["unresolved_corrupt"]:
        raise RuntimeError("Reserved-draw feature cache is incomplete")
    out_path = output_root(config) / "data" / "pilot_feature_audit.json"
    write_json(out_path, audit)
    sign_file(out_path)
    return out_path
