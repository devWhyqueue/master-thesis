"""TCGA-UT participant identity, pre-extracted feature manifest, and case splits.

Unlike BRACS/CAMELYON16/PANDA, TCGA-UT patch features are pre-extracted on the
cluster as chunked ``cls_patchmean`` tensors; this adapter matches those feature
chunks to their raw class-folder labels rather than tiling images itself.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd


def tcga_case_id(slide_id: str) -> str:
    """Return the participant barcode encoded in a TCGA slide identifier."""
    parts = slide_id.split("-")
    if len(parts) >= 3 and parts[0] == "TCGA":
        return "-".join(parts[:3])
    return slide_id


def collect_slide_labels(raw_root: Path) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Collect slide-to-class mapping from raw TCGA-UT class/split/slide folders."""
    labels: dict[str, str] = {}
    conflicts: dict[str, list[str]] = defaultdict(list)
    if not raw_root.exists():
        raise FileNotFoundError(f"Raw TCGA-UT root does not exist: {raw_root}")
    for class_dir in sorted(path for path in raw_root.iterdir() if path.is_dir()):
        class_name = class_dir.name
        for split_dir in sorted(path for path in class_dir.iterdir() if path.is_dir()):
            for slide_entry in sorted(split_dir.iterdir()):
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


def assign_class_splits(
    units: list[str], seed: int, val_fraction: float = 0.15, test_fraction: float = 0.15
) -> dict[str, str]:
    """Assign per-class split units (participants) to train/validation/test."""
    rng = np.random.default_rng(seed)
    shuffled = list(units)
    rng.shuffle(shuffled)
    n_units = len(shuffled)
    if n_units == 1:
        return {shuffled[0]: "train"}
    if n_units == 2:
        return {shuffled[0]: "train", shuffled[1]: "test"}
    n_test = max(1, int(round(n_units * test_fraction)))
    n_val = max(1, int(round(n_units * val_fraction)))
    if n_test + n_val >= n_units:
        n_test, n_val = 1, 1
    assignments: dict[str, str] = {}
    for case_id in shuffled[:n_test]:
        assignments[case_id] = "test"
    for case_id in shuffled[n_test : n_test + n_val]:
        assignments[case_id] = "validation"
    for case_id in shuffled[n_test + n_val :]:
        assignments[case_id] = "train"
    return assignments


def split_cases(slide_manifest: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Return participant-disjoint split assignments keyed by case_id, per class."""
    rows: list[dict[str, str]] = []
    for _, class_df in slide_manifest.groupby("cancer_type"):
        assignments = assign_class_splits(
            sorted(class_df["case_id"].astype(str).unique()), seed
        )
        rows.extend(
            {"case_id": case_id, "split": split}
            for case_id, split in assignments.items()
        )
    return pd.DataFrame(rows)


def assert_case_disjoint(frame: pd.DataFrame) -> None:
    """Raise if any participant appears in more than one split."""
    split_counts = cast(pd.Series, frame.groupby("case_id")["split"].nunique())
    leaking = [
        str(case_id) for case_id, count in split_counts.items() if int(count) > 1
    ]
    if leaking:
        raise ValueError(f"TCGA-UT participant leakage: {leaking[:5]}")
