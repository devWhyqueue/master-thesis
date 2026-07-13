from __future__ import annotations

import pandas as pd
import torch

from imbalance_benchmark.datasets.tcga_ut import (
    assert_case_disjoint,
    assign_class_splits,
    build_feature_manifest,
    collect_slide_labels,
    split_cases,
    strip_feature_suffix,
    tcga_case_id,
)


def test_tcga_case_id_extracts_participant_barcode() -> None:
    assert tcga_case_id("TCGA-AB-1234-01Z-00-DX1") == "TCGA-AB-1234"
    assert tcga_case_id("not-a-tcga-id") == "not-a-tcga-id"


def test_strip_feature_suffix_removes_chunk_index() -> None:
    assert strip_feature_suffix("TCGA-AB-1234-01Z_3", "_[0-9]+") == "TCGA-AB-1234-01Z"


def test_collect_slide_labels_maps_class_folders(tmp_path) -> None:
    for cls, split, slide in (("LUAD", "train", "slideA"), ("LUSC", "train", "slideB")):
        (tmp_path / cls / split / slide).mkdir(parents=True)

    labels, conflicts = collect_slide_labels(tmp_path)

    assert labels == {"slideA": "LUAD", "slideB": "LUSC"}
    assert conflicts == {}


def test_collect_slide_labels_reports_conflicts(tmp_path) -> None:
    (tmp_path / "LUAD" / "train" / "slideA").mkdir(parents=True)
    (tmp_path / "LUSC" / "train" / "slideA").mkdir(parents=True)

    labels, conflicts = collect_slide_labels(tmp_path)

    assert labels["slideA"] == "LUAD"
    assert conflicts["slideA"] == ["LUAD", "LUSC"]


def test_build_feature_manifest_matches_chunks_to_labels(tmp_path) -> None:
    torch.save(torch.randn(30, 2560), tmp_path / "TCGA-AB-0001-01Z_0.pt")
    torch.save(torch.randn(5, 2560), tmp_path / "TCGA-AB-0001-01Z_1.pt")
    torch.save(torch.randn(30, 2560), tmp_path / "unlabelled-slide_0.pt")
    labels = {"TCGA-AB-0001-01Z": "LUAD"}

    manifest, slide_manifest, unmatched = build_feature_manifest(tmp_path, labels)

    assert len(manifest) == 2
    assert slide_manifest.loc[0, "n_feature_chunks"] == 2
    assert slide_manifest.loc[0, "case_id"] == "TCGA-AB-0001"
    assert unmatched == [str(tmp_path / "unlabelled-slide_0.pt")]


def test_assign_class_splits_covers_all_units_without_overlap() -> None:
    units = [f"case_{i}" for i in range(20)]
    assignments = assign_class_splits(units, seed=0)

    assert set(assignments) == set(units)
    assert set(assignments.values()) <= {"train", "validation", "test"}


def test_split_cases_are_case_disjoint() -> None:
    slide_manifest = pd.DataFrame(
        {
            "slide_id": [f"s{i}" for i in range(12)],
            "case_id": [f"case_{i}" for i in range(12)],
            "cancer_type": ["LUAD"] * 6 + ["LUSC"] * 6,
        }
    )
    assigned = slide_manifest.merge(split_cases(slide_manifest, 0), on="case_id", how="inner")

    assert_case_disjoint(assigned)
    assert set(assigned["split"]).issubset({"train", "validation", "test"})
