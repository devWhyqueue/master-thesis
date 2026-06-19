from __future__ import annotations

from pathlib import Path

import pandas as pd

from tcga_ut_imbalanced.progan import (
    combined_manifest,
    resolve_real_patch_path,
    tail_classes,
    with_real_metadata,
)


def test_tail_classes_selects_minority_train_classes() -> None:
    manifest = pd.DataFrame(
        [
            {"split": "train", "cancer_type": "a", "slide_id": "a0", "patch_id": "0", "feature_path": "a.pt", "feature_index": 0},
            {"split": "train", "cancer_type": "a", "slide_id": "a1", "patch_id": "1", "feature_path": "a.pt", "feature_index": 1},
            {"split": "train", "cancer_type": "b", "slide_id": "b0", "patch_id": "0", "feature_path": "b.pt", "feature_index": 0},
            {"split": "validation", "cancer_type": "c", "slide_id": "c0", "patch_id": "0", "feature_path": "c.pt", "feature_index": 0},
        ]
    )

    assert tail_classes(manifest) == ["b"]


def test_combined_manifest_keeps_real_rows_and_tags_synthetic_cache() -> None:
    manifest = pd.DataFrame(
        [
            {"split": "train", "slide_id": "real", "cancer_type": "a", "patch_id": "0", "feature_path": "real.pt", "feature_index": 4},
            {"split": "validation", "slide_id": "val", "cancer_type": "a", "patch_id": "1", "feature_path": "val.pt", "feature_index": 5},
        ]
    )
    generated = {
        25: pd.DataFrame(
            [
                {
                    "split": "train",
                    "slide_id": "syn-a",
                    "cancer_type": "a",
                    "patch_id": "syn-a",
                    "image_path": "syn-a.jpg",
                    "is_synthetic": True,
                    "final_depth_epochs": 25,
                }
            ]
        )
    }

    combined = combined_manifest(manifest, generated, "progan_cache.pt")

    assert len(combined) == 3
    assert not bool(combined.iloc[0]["is_synthetic"])
    synthetic = combined.iloc[-1]
    assert synthetic["feature_path"] == "progan_cache.pt"
    assert int(synthetic["feature_index"]) == 0
    assert int(synthetic["final_depth_epochs"]) == 25


def test_with_real_metadata_marks_all_rows_as_non_synthetic() -> None:
    manifest = pd.DataFrame(
        [{"split": "train", "slide_id": "real", "cancer_type": "a", "patch_id": "0", "feature_path": "real.pt", "feature_index": 4}]
    )

    tagged = with_real_metadata(manifest)

    assert tagged["is_synthetic"].tolist() == [False]
    assert tagged["final_depth_epochs"].tolist() == [0]


def test_resolve_real_patch_path_uses_tcga_layout(tmp_path: Path) -> None:
    expected = tmp_path / "a" / "0" / "slide-1" / "1_2.jpg"
    expected.parent.mkdir(parents=True)
    expected.write_bytes(b"x")

    assert resolve_real_patch_path(tmp_path, "0", "a", "slide-1", "1_2") == expected
