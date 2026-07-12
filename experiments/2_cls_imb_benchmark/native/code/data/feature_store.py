"""Resolve chunked cls_patchmean Virchow2 tensors keyed by slide and patch order."""

from __future__ import annotations

import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import cast

import pandas as pd
import torch

from data.sampling import patch_sort_key

DEFAULT_FEATURE_DIR = (
    "/home/space/datasets/patho_ds/tcga-ut/patch_features/"
    "cls_patchmean/virchow_virchow2/raw"
)
DEFAULT_SUFFIX_PATTERN = "_[0-9]+$"
PATCHES_PER_CHUNK = 30


@lru_cache(maxsize=512)
def _ordered_patch_indices(patch_ids: tuple[str, ...]) -> dict[str, int]:
    ordered = sorted(patch_ids, key=patch_sort_key)
    return {patch: index for index, patch in enumerate(ordered)}


class SlideFeatureStore:
    """Index cls_patchmean chunk files and map manifest patches to row indices."""

    def __init__(
        self,
        feature_dir: str,
        suffix_pattern: str = DEFAULT_SUFFIX_PATTERN,
        patches_per_chunk: int = PATCHES_PER_CHUNK,
    ) -> None:
        self.feature_dir = Path(feature_dir)
        if not self.feature_dir.is_dir():
            raise FileNotFoundError(f"Feature directory not found: {feature_dir}")
        self.suffix_pattern = suffix_pattern
        self.patches_per_chunk = patches_per_chunk
        self._chunks_by_slide = _build_slide_chunk_paths(
            self.feature_dir, suffix_pattern
        )

    def resolve_patch(
        self, slide_id: str, patch_ids: list[str], patch_id: str
    ) -> tuple[str, int]:
        """Return the chunk path and row index for one manifest patch."""
        patch_index = self.patch_index(patch_ids, patch_id)
        chunk_index = patch_index // self.patches_per_chunk
        row_index = patch_index % self.patches_per_chunk
        chunk_path = self._chunks_by_slide[slide_id][chunk_index]
        return str(chunk_path), row_index

    def patch_index(self, patch_ids: list[str], patch_id: str) -> int:
        """Return the row index for one patch after deterministic sorting."""
        return _ordered_patch_indices(tuple(patch_ids))[patch_id]

    def load_patch_feature(
        self, slide_id: str, patch_ids: list[str], patch_id: str
    ) -> torch.Tensor:
        """Load one patch embedding from the matching chunk tensor."""
        path, index = self.resolve_patch(slide_id, patch_ids, patch_id)
        return load_feature_row(path, index)


@lru_cache(maxsize=512)
def load_slide_features(path: str) -> torch.Tensor:
    """Load a feature tensor and normalize to (n_instances, dim)."""
    tensor = torch.load(path, map_location="cpu")
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected tensor at {path}.")
    features = tensor.float()
    if features.ndim == 1:
        return features.unsqueeze(0)
    if features.ndim > 2:
        return features.reshape(-1, features.shape[-1])
    return features


def load_feature_row(path: str, index: int | None = None) -> torch.Tensor:
    """Load one feature vector without silent aggregation."""
    features = load_slide_features(path)
    if index is not None:
        return features[int(index)].squeeze()
    if features.shape[0] == 1:
        return features[0].squeeze()
    raise ValueError(
        f"Feature file {path} has {features.shape[0]} rows; "
        "provide feature_index for multi-row tensors."
    )


def verify_feature_store(
    feature_dir: str, expected_dim: int = 2560
) -> dict[str, object]:
    """Inspect one chunk tensor and return a short verification report."""
    store = SlideFeatureStore(feature_dir)
    sample_slide = next(iter(store._chunks_by_slide))
    sample_path = str(store._chunks_by_slide[sample_slide][0])
    tensor = load_slide_features(sample_path)
    feature_dim = int(tensor.shape[-1])
    return {
        "feature_dir": feature_dir,
        "sample_slide_id": sample_slide,
        "sample_path": sample_path,
        "shape": list(tensor.shape),
        "feature_dim": feature_dim,
        "expected_dim": expected_dim,
        "dim_matches": feature_dim == expected_dim,
        "n_slides_indexed": len(store._chunks_by_slide),
        "layout": "chunked_slide_tensors",
        "patches_per_chunk": store.patches_per_chunk,
    }


def enrich_frame_with_features(
    frame: pd.DataFrame, store: SlideFeatureStore
) -> pd.DataFrame:
    """Return a row-level frame with feature_path and feature_index columns."""
    if "patch_ids" not in frame.columns:
        return _enrich_row_level(frame, store)
    rows: list[dict[str, object]] = []
    for _, row in frame.iterrows():
        rows.extend(_patch_feature_row(row, store))
    return pd.DataFrame(rows)


def _patch_feature_row(
    row: "pd.Series", store: SlideFeatureStore
) -> list[dict[str, object]]:
    slide_id = str(row["slide_id"])
    patch_ids = [str(patch_id) for patch_id in row["patch_ids"]]
    optional: dict[str, object] = {}
    for column in ("case_id", "sample_instance"):
        if column not in row.index:
            continue
        value = row.at[column]
        if pd.isna(value):
            continue
        optional[column] = value
    rows_out: list[dict[str, object]] = []
    for patch_id in patch_ids:
        feature_path, feature_index = store.resolve_patch(slide_id, patch_ids, patch_id)
        rows_out.append(
            {
                **optional,
                "slide_id": slide_id,
                "cancer_type": str(row["cancer_type"]),
                "patch_id": patch_id,
                "feature_path": feature_path,
                "feature_index": feature_index,
            }
        )
    return rows_out


def expand_splits_for_wsi_manifest(
    frame_by_split: dict[str, pd.DataFrame],
    feature_dir: str,
) -> pd.DataFrame:
    """Expand capped split frames into one WSI-ready manifest_splits table."""
    store = SlideFeatureStore(feature_dir)
    parts = []
    for split, frame in frame_by_split.items():
        enriched = enrich_frame_with_features(frame, store)
        tagged = enriched.copy()
        tagged["split"] = split
        parts.append(tagged)
    return cast(pd.DataFrame, pd.concat(parts, ignore_index=True))


def _enrich_row_level(frame: pd.DataFrame, store: SlideFeatureStore) -> pd.DataFrame:
    enriched = frame.copy()
    if "feature_path" not in enriched.columns:
        enriched[["feature_path", "feature_index"]] = enriched.apply(
            lambda row: pd.Series(
                store.resolve_patch(
                    str(row["slide_id"]),
                    [str(row["patch_id"])],
                    str(row["patch_id"]),
                )
            ),
            axis=1,
        )
    return enriched


def maybe_feature_store(feature_path: str) -> SlideFeatureStore | None:
    """Return a feature store when feature_path is a feature directory."""
    path = Path(feature_path) if feature_path else None
    if path is None or path.is_file() or not path.is_dir():
        return None
    try:
        return SlideFeatureStore(str(path))
    except (FileNotFoundError, RuntimeError):
        return None


def _build_slide_chunk_paths(
    feature_dir: Path, suffix_pattern: str
) -> dict[str, list[Path]]:
    grouped: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    for path in sorted(feature_dir.glob("*.pt")):
        stem = path.stem
        slide_id = re.sub(f"{suffix_pattern}$", "", stem)
        suffix = stem[len(slide_id) :]
        chunk_index = int(suffix.lstrip("_")) if suffix else 0
        grouped[slide_id].append((chunk_index, path))
    if not grouped:
        raise RuntimeError(f"No .pt features found under {feature_dir}")
    return {
        slide_id: [path for _, path in sorted(items, key=lambda item: item[0])]
        for slide_id, items in grouped.items()
    }
