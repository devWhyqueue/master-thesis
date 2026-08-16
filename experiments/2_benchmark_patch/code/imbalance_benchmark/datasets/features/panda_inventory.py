"""Signed PANDA feature-cache inventory reduction and verification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from imbalance_benchmark.common import (
    compute_sha256,
    sign_file,
    verify_signed_file,
    write_json,
)
from imbalance_benchmark.datasets.data.panda.publish import (
    LOCKED_BENIGN_PATCHES,
    LOCKED_CANCER_PATCHES,
    LOCKED_LABELLED_SLIDES,
    LOCKED_SLIDES,
)
from imbalance_benchmark.datasets.feature_provenance import (
    FEATURE_DIM,
    load_stored_feature_tensor,
    resolve_feature_provenance,
)
from imbalance_benchmark.datasets.features.attach import validate_cached_features
from imbalance_benchmark.datasets.features.cache_manifest import (
    cache_records,
    merge_pending_slides,
)


def reduce_feature_inventory(
    config: dict[str, Any], frame: pd.DataFrame, root: Path
) -> None:
    """Serially merge worker records and sign complete tensor inventory."""
    expected = {
        str(slide): (root / f"{slide}.pt", _slide_identity(group))
        for slide, group in frame.groupby("slide_id", sort=False)
    }
    merge_pending_slides(root, expected)
    validate_cached_features(frame, root)
    provenance = resolve_feature_provenance(config["feature_extraction"])
    records = cache_records(root)
    _validate_tensors(root, records)
    path = Path(config["feature_inventory_path"])
    write_json(
        path,
        {
            "feature_provenance": provenance,
            "cache_manifest_sha256": compute_sha256(
                root / "feature_cache_manifest.json"
            ),
            "slides": records,
            "slide_count": len(records),
            "patch_count": len(frame),
        },
    )
    sign_file(path)


def verify_feature_inventory(
    config: dict[str, Any], frame: pd.DataFrame, root: Path
) -> None:
    """Refuse final prepare if signed feature source differs or is incomplete."""
    path = Path(config["feature_inventory_path"])
    verify_signed_file(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    provenance = resolve_feature_provenance(config["feature_extraction"])
    valid = (
        payload.get("feature_provenance") == provenance
        and payload.get("cache_manifest_sha256")
        == compute_sha256(root / "feature_cache_manifest.json")
        and payload.get("slide_count") == frame.slide_id.nunique()
        and payload.get("patch_count") == len(frame)
    )
    if not valid:
        raise ValueError("PANDA feature inventory coverage or provenance differs")
    validate_cached_features(frame, root)
    _validate_tensors(root, cache_records(root))


def load_materialized_inventory(dataset: dict[str, Any]) -> pd.DataFrame:
    """Load signed materialized metadata and recheck locked cohort totals."""
    path = Path(dataset["canonical_inventory_path"])
    sidecar_path = Path(dataset["materialization_sidecar"])
    verify_signed_file(path)
    verify_signed_file(sidecar_path)
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if sidecar.get("inventory_sha256") != compute_sha256(path):
        raise ValueError(
            "PANDA canonical inventory differs from signed materialization"
        )
    frame = pd.read_csv(path)
    _validate_materialized_shape(frame, sidecar)
    return frame


def _validate_materialized_shape(frame: pd.DataFrame, sidecar: dict[str, Any]) -> None:
    required = {
        "slide_id",
        "case_id",
        "patch_id",
        "patch_label",
        "image_path",
        "shard_index",
    }
    counts = frame.patch_label.value_counts()
    expected = {
        "official_slides": LOCKED_SLIDES,
        "labelled_slides": LOCKED_LABELLED_SLIDES,
        "benign_patches": LOCKED_BENIGN_PATCHES,
        "cancer_patches": LOCKED_CANCER_PATCHES,
    }
    actual = {
        "official_slides": sidecar.get("cohort_counts", {}).get("official_slides"),
        "labelled_slides": frame.loc[
            frame.patch_label.isin(("benign", "cancer")), "slide_id"
        ].nunique(),
        "benign_patches": int(counts.get("benign", 0)),
        "cancer_patches": int(counts.get("cancer", 0)),
    }
    if (
        required - set(frame)
        or frame.duplicated("patch_id").any()
        or actual != expected
        or sidecar.get("cohort_counts") != expected
    ):
        raise ValueError(
            "PANDA canonical inventory is invalid or counts differ from protocol"
        )


def _slide_identity(group: pd.DataFrame) -> list[str]:
    return [
        f"{patch}\0{image}"
        for patch, image in zip(
            group.patch_id.astype(str), group.image_path.astype(str), strict=True
        )
    ]


def _validate_tensors(root: Path, records: dict[str, dict[str, object]]) -> None:
    for slide in records:
        tensor = load_stored_feature_tensor(str(root / f"{slide}.pt"))
        if tensor.ndim != 2 or tensor.shape[1] != FEATURE_DIM:
            raise ValueError(f"PANDA feature dimension differs for {slide}")
        if tensor.dtype != torch.float32:
            raise ValueError(f"PANDA feature dtype differs for {slide}")
