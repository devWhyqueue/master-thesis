"""TCGA-UT patch regime: image-backed manifest from the materialized SqFS.

Reads every image directly from the Zenodo-authenticated, project-owned SqFS
built by :mod:`imbalance_benchmark.datasets.tcga_ut.pack`, letting the
standard Virchow2 extraction pipeline (``datasets/features/attach.py``)
extract features the same way it does for BRACS/CAMELYON16/PANDA.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pandas as pd

from imbalance_benchmark.common import verify_signed_file
from imbalance_benchmark.datasets.tcga_ut.source import ZENODO_RECORD_ID, ZENODO_VERSION
from imbalance_benchmark.datasets.tcga_ut.splits import (
    assert_case_disjoint,
    split_cases,
    tcga_case_id,
)

__all__ = [
    "validate_source_provenance",
    "iter_class_slide_images",
    "build_image_rows",
    "validate_image_cohort",
    "build_image_manifest",
]


def validate_source_provenance(dataset: dict[str, Any]) -> dict[str, Any]:
    """Verify the signed materialization sidecar backing the image-backed manifest."""
    sidecar_path = Path(dataset["materialization_sidecar"])
    verify_signed_file(sidecar_path)
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if not payload.get("validated"):
        raise ValueError("TCGA-UT source materialization was not validated")
    if payload.get("zenodo_record_id") != ZENODO_RECORD_ID:
        raise ValueError("TCGA-UT source materialization used the wrong Zenodo record")
    if payload.get("zenodo_version") != ZENODO_VERSION:
        raise ValueError("TCGA-UT source materialization used the wrong Zenodo version")
    return payload


def _leaf_image_dirs(class_dir: Path) -> Iterator[tuple[Path, list[Path]]]:
    """Yield every directory under ``class_dir`` that directly holds JPGs (a slide)."""
    for dirpath, _dirnames, filenames in os.walk(class_dir):
        images = sorted(
            Path(dirpath) / name for name in filenames if name.lower().endswith(".jpg")
        )
        if images:
            yield Path(dirpath), images


def iter_class_slide_images(root: Path) -> Iterator[tuple[str, str, Path]]:
    """Yield ``(class_name, slide_id, image_path)`` for every JPG under ``root``.

    A slide is any directory that directly contains JPGs; its class is the
    top-level directory under ``root`` that contains it.
    """
    if not root.exists():
        raise FileNotFoundError(f"TCGA-UT image root does not exist: {root}")
    for class_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for slide_dir, images in _leaf_image_dirs(class_dir):
            for image in images:
                yield class_dir.name, slide_dir.name, image


def build_image_rows(root: Path) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Enumerate every source image into per-patch rows, and any label conflicts."""
    labels: dict[str, str] = {}
    conflicts: dict[str, list[str]] = defaultdict(list)
    records: list[dict[str, str]] = []
    for class_name, slide_id, image_path in iter_class_slide_images(root):
        existing = labels.get(slide_id)
        if existing is None:
            labels[slide_id] = class_name
        elif existing != class_name:
            conflicts[slide_id].extend([existing, class_name])
        records.append(
            {
                "dataset": "tcga_ut",
                "case_id": tcga_case_id(slide_id),
                "slide_id": slide_id,
                "cancer_type": class_name,
                "patch_id": f"{slide_id}/{image_path.stem}",
                "image_path": str(image_path),
            }
        )
    return pd.DataFrame(records), dict(conflicts)


def validate_image_cohort(
    frame: pd.DataFrame,
    conflicts: dict[str, list[str]],
    expected_patch_count: int,
    expected_class_count: int,
    expected_slide_count: int,
) -> None:
    """Require the configured image cohort to match the locked cohort exactly."""
    if conflicts:
        raise ValueError(f"TCGA-UT label conflicts: {sorted(conflicts)[:5]}")
    duplicate_ids = cast(
        pd.Series, frame["patch_id"][frame["patch_id"].duplicated()]
    ).unique()
    if len(duplicate_ids):
        raise ValueError(
            f"TCGA-UT duplicate patch identities: {sorted(duplicate_ids)[:5]}"
        )
    if len(frame) != expected_patch_count:
        raise ValueError("TCGA-UT patch count differs from the locked cohort")
    if frame["cancer_type"].nunique() != expected_class_count:
        raise ValueError("TCGA-UT class count differs from the locked cohort")
    if frame["slide_id"].nunique() != expected_slide_count:
        raise ValueError("TCGA-UT slide count differs from the locked cohort")


def build_image_manifest(config: dict[str, Any]) -> pd.DataFrame:
    """Build and validate the definitive image-backed TCGA-UT patch-row manifest."""
    dataset = config["dataset"]
    validate_source_provenance(dataset)
    frame, conflicts = build_image_rows(Path(dataset["root"]))
    validate_image_cohort(
        frame,
        conflicts,
        int(dataset["expected_patch_count"]),
        int(dataset["expected_class_count"]),
        int(dataset["expected_slide_count"]),
    )
    slides = cast(
        pd.DataFrame,
        frame.drop_duplicates("slide_id")[["slide_id", "case_id", "cancer_type"]],
    )
    assignment = split_cases(slides, int(dataset.get("seed", 0)))
    tagged = frame.merge(assignment, on="case_id", how="inner")
    assert_case_disjoint(tagged)
    return tagged
