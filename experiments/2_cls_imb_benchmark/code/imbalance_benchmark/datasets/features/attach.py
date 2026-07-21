from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch

from imbalance_benchmark.datasets import features as feature_lib
from imbalance_benchmark.datasets.features.cache import load_slide_features
from imbalance_benchmark.datasets.feature_provenance import (
    VIRCHOW2_REVISION,
    VIRCHOW2_WEIGHTS_SHA256,
    record_cached_slide,
    resolve_feature_provenance,
    validate_cached_slide,
    validate_feature_cache,
)

__all__ = ["attach_extracted_features"]


def attach_extracted_features(
    frame: pd.DataFrame,
    feature_root: Path,
    model_name: str = "hf-hub:paige-ai/Virchow2",
    batch_size: int = 64,
    dtype: str = "float16",
    device: torch.device | None = None,
    revision: str = VIRCHOW2_REVISION,
    weights_sha256: str = VIRCHOW2_WEIGHTS_SHA256,
) -> pd.DataFrame:
    """Extract one stacked per-slide feature tensor per slide and attach references.

    Rows must already be in the deterministic per-slide patch order fixed by the
    dataset adapter. Existing ``<feature_root>/<slide_id>.pt`` files are reused
    rather than re-extracted. All slides in one call share a single loaded
    encoder instance instead of reloading it per slide.
    """
    options = {
        "model_name": model_name,
        "batch_size": batch_size,
        "dtype": dtype,
        "device": device,
        "revision": revision,
        "weights_sha256": weights_sha256,
    }
    _prepare_feature_cache(feature_root, options)
    enriched = frame.copy()
    feature_paths, feature_indices = _feature_references(
        enriched, feature_root, options
    )
    enriched["feature_path"] = feature_paths
    enriched["feature_index"] = feature_indices.astype(int)
    return enriched


def _prepare_feature_cache(feature_root: Path, options: dict[str, Any]) -> None:
    feature_root.mkdir(parents=True, exist_ok=True)
    provenance = resolve_feature_provenance(options)
    validate_feature_cache(feature_root, provenance)


def _feature_references(
    frame: pd.DataFrame, feature_root: Path, options: dict[str, Any]
) -> tuple[pd.Series, pd.Series]:
    feature_paths = pd.Series(index=frame.index, dtype=object)
    feature_indices = pd.Series(index=frame.index, dtype=object)
    model_cache: dict[str, Any] = {}
    for slide_id, group in frame.groupby("slide_id", sort=False):
        slide_path = _ensure_slide_features(
            group, feature_root, str(slide_id), options, model_cache
        )
        feature_paths.loc[group.index] = str(slide_path)
        feature_indices.loc[group.index] = range(len(group))
    return feature_paths, feature_indices


def _ensure_slide_features(
    group: pd.DataFrame,
    feature_root: Path,
    slide_id: str,
    options: dict[str, Any],
    model_cache: dict[str, Any],
) -> Path:
    slide_path = feature_root / f"{slide_id}.pt"
    identities = _ordered_patch_identity(group)
    if slide_path.exists():
        validate_cached_slide(
            feature_root,
            slide_id,
            slide_path,
            identities,
            len(load_slide_features(str(slide_path))),
        )
        return slide_path
    image_paths = group["image_path"].astype(str).tolist()
    tensor = feature_lib.extract_slide_features(
        image_paths,
        str(options["model_name"]),
        int(options["batch_size"]),
        str(options["dtype"]),
        options["device"],
        str(options["revision"]),
        str(options["weights_sha256"]),
        model_cache=model_cache,
    )
    torch.save(tensor, slide_path)
    record_cached_slide(feature_root, slide_id, slide_path, identities, len(tensor))
    return slide_path


def _ordered_patch_identity(group: pd.DataFrame) -> list[str]:
    """Return the exact patch identity sequence tied to cached tensor rows."""
    patch_ids = (
        group["patch_id"].astype(str)
        if "patch_id" in group
        else group["image_path"].astype(str)
    )
    return [
        f"{patch_id}\0{image_path}"
        for patch_id, image_path in zip(
            patch_ids, group["image_path"].astype(str), strict=True
        )
    ]
