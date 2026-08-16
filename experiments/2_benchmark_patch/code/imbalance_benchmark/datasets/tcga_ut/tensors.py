"""TCGA-UT WSI regime: pre-extracted ``cls_patchmean`` tensor-chunk manifest.

Patch features are pre-extracted on the cluster as chunked tensors; this
adapter matches those feature chunks to their raw class-folder labels rather
than tiling images itself. The patch regime instead reads images directly;
see :mod:`imbalance_benchmark.datasets.tcga_ut.image`.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from imbalance_benchmark.datasets.feature_provenance import (
    resolve_feature_provenance,
    validate_preextracted_features,
)
from imbalance_benchmark.datasets.tcga_ut.splits import (
    assert_case_disjoint,
    split_cases,
    tcga_case_id,
)

__all__ = [
    "collect_slide_labels",
    "strip_feature_suffix",
    "build_feature_manifest",
    "validate_feature_coverage",
    "build_manifest",
]


def collect_slide_labels(raw_root: Path) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Collect slide-to-class mapping from raw TCGA-UT class/split/slide folders."""
    labels: dict[str, str] = {}
    conflicts: dict[str, list[str]] = defaultdict(list)
    if not raw_root.exists():
        raise FileNotFoundError(f"Raw TCGA-UT root does not exist: {raw_root}")
    for class_dir in sorted(path for path in raw_root.iterdir() if path.is_dir()):
        class_name = class_dir.name
        for split_dir in sorted(path for path in class_dir.iterdir() if path.is_dir()):
            # A slide is a non-empty directory of patches; skip stray empty dirs
            # (the shared cluster tree has junk folds like `40.000000/2`).
            for slide_entry in sorted(
                p for p in split_dir.iterdir() if p.is_dir() and any(p.iterdir())
            ):
                slide_id = slide_entry.name
                existing = labels.get(slide_id)
                if existing is None:
                    labels[slide_id] = class_name
                elif existing != class_name:
                    conflicts[slide_id].extend([existing, class_name])
    return labels, conflicts


def strip_feature_suffix(feature_stem: str, suffix_pattern: str) -> str:
    """Remove feature chunk suffix from a feature identifier."""
    return re.sub(f"{suffix_pattern}$", "", feature_stem)


def _match_feature_chunk(
    feature_path: Path, labels: dict[str, str], suffix_pattern: str
) -> dict[str, str] | None:
    """Match one feature chunk file to its raw label, or None if unmatched."""
    feature_id = feature_path.stem
    slide_id = strip_feature_suffix(feature_id, suffix_pattern)
    class_name = labels.get(slide_id)
    if class_name is None:
        return None
    return {
        "feature_id": feature_id,
        "slide_id": slide_id,
        "case_id": tcga_case_id(slide_id),
        "cancer_type": class_name,
        "feature_path": str(feature_path),
    }


def build_feature_manifest(
    feature_dir: Path,
    labels: dict[str, str],
    feature_glob: str = "*.pt",
    suffix_pattern: str = "_[0-9]+",
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Match feature chunk files to raw labels and return chunk/slide manifests.

    Returns (chunk_level_manifest, slide_level_manifest, unmatched_feature_paths).
    """
    matches = [
        (path, _match_feature_chunk(path, labels, suffix_pattern))
        for path in sorted(feature_dir.glob(feature_glob))
    ]
    unmatched = [str(path) for path, row in matches if row is None]
    manifest = pd.DataFrame([row for _, row in matches if row is not None])
    if manifest.empty:
        raise RuntimeError("No feature files could be matched to TCGA-UT labels.")
    slide_manifest = _slide_manifest(manifest)
    return manifest, slide_manifest, unmatched


def validate_feature_coverage(
    labels: dict[str, str],
    conflicts: dict[str, list[str]],
    slide_manifest: pd.DataFrame,
    unmatched: list[str],
    expected_slides: int,
    expected_classes: int,
) -> None:
    """Require the configured raw and feature cohorts to correspond exactly."""
    if conflicts:
        raise ValueError(f"TCGA-UT label conflicts: {sorted(conflicts)[:5]}")
    if unmatched:
        raise ValueError(f"TCGA-UT unmatched feature chunks: {unmatched[:5]}")
    feature_slides = set(slide_manifest["slide_id"].astype(str))
    missing = sorted(set(labels) - feature_slides)
    if missing:
        raise ValueError(f"TCGA-UT raw slides without features: {missing[:5]}")
    if len(labels) != expected_slides or len(feature_slides) != expected_slides:
        raise ValueError("TCGA-UT slide count differs from the locked cohort")
    if len(set(labels.values())) != expected_classes:
        raise ValueError("TCGA-UT class count differs from the locked cohort")


def build_manifest(config: dict[str, Any]) -> pd.DataFrame:
    """Build and validate the definitive TCGA-UT patch-row manifest."""
    dataset = config["dataset"]
    labels, conflicts = collect_slide_labels(Path(dataset["raw_root"]))
    chunks, slides, unmatched = build_feature_manifest(
        Path(dataset["feature_dir"]),
        labels,
        str(dataset.get("feature_glob", "*.pt")),
        str(dataset.get("feature_suffix_pattern", "_[0-9]+")),
    )
    validate_feature_coverage(
        labels,
        conflicts,
        slides,
        unmatched,
        int(dataset["expected_slide_count"]),
        int(dataset["expected_class_count"]),
    )
    row_counts = _validate_provenance(config, dataset, chunks)
    frame = _expand_chunks(chunks, row_counts)
    if len(frame) != int(dataset["expected_patch_count"]):
        raise ValueError("TCGA-UT patch count differs from the locked cohort")
    assignment = split_cases(slides, int(dataset.get("seed", 0)))
    tagged = frame.merge(assignment, on="case_id", how="inner")
    assert_case_disjoint(tagged)
    return tagged


def _validate_provenance(
    config: dict[str, Any], dataset: dict[str, Any], chunks: pd.DataFrame
) -> dict[str, int]:
    feature_config = config.get("feature_extraction", {})
    if not isinstance(feature_config, dict):
        raise ValueError("feature_extraction config must be a mapping")
    return validate_preextracted_features(
        Path(dataset["feature_provenance_manifest"]),
        [Path(path) for path in chunks["feature_path"].astype(str)],
        resolve_feature_provenance(feature_config),
    )


def _expand_chunks(chunks: pd.DataFrame, row_counts: dict[str, int]) -> pd.DataFrame:
    counts = chunks["feature_path"].astype(str).map(row_counts.__getitem__)
    expanded = chunks.loc[chunks.index.repeat(counts)].reset_index(drop=True)
    expanded["dataset"] = "tcga_ut"
    expanded["feature_index"] = expanded.groupby("feature_path", sort=False).cumcount()
    return expanded[
        [
            "dataset",
            "case_id",
            "slide_id",
            "cancer_type",
            "feature_path",
            "feature_index",
        ]
    ]


def _slide_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    grouped = manifest.groupby(
        ["slide_id", "case_id", "cancer_type"], as_index=False
    ).agg(n_feature_chunks=("feature_id", "count"))
    rows = sorted(
        (
            {
                "slide_id": str(row["slide_id"]),
                "case_id": str(row["case_id"]),
                "cancer_type": str(row["cancer_type"]),
                "n_feature_chunks": int(row["n_feature_chunks"]),
            }
            for _, row in grouped.iterrows()
        ),
        key=lambda row: (row["cancer_type"], row["slide_id"]),
    )
    return pd.DataFrame(rows)
