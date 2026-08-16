from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch

from imbalance_benchmark.datasets import build_manifest
from imbalance_benchmark.datasets.data import BagFeatureDataset
from imbalance_benchmark.datasets.data import ImbalanceDataset
from imbalance_benchmark.manifest.construction_helpers import class_support_counts
from imbalance_benchmark.manifest.statistics import support_statistics

def test_slide_statistics_count_mixed_label_slides_once_per_class() -> None:
    rows = pd.DataFrame(
        [
            {"slide_id": "s1", "cancer_type": "normal"},
            {"slide_id": "s1", "cancer_type": "tumor"},
            {"slide_id": "s2", "cancer_type": "normal"},
        ]
    )

    statistics = support_statistics(rows)

    assert statistics["patch"]["counts"] == {"normal": 2, "tumor": 1}
    assert statistics["slide"]["counts"] == {"normal": 2, "tumor": 1}

def test_build_manifest_rejects_unknown_dataset_name() -> None:
    with pytest.raises(ValueError, match="Unknown dataset"):
        build_manifest({"dataset": {"name": "not-a-real-dataset"}})

def test_dataset_uses_the_locked_global_class_index_even_when_a_split_is_sparse(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.csv"
    feature = tmp_path / "feature.pt"
    torch.save(torch.ones(1, 2), feature)
    pd.DataFrame(
        [
            {
                "case_id": "p1",
                "slide_id": "s1",
                "cancer_type": "A",
                "feature_path": feature,
                "split": "validation",
            }
        ]
    ).to_csv(manifest, index=False)

    dataset = ImbalanceDataset(manifest, "validation", class_names=["A", "B"])

    assert dataset.classes == ["A", "B"]
    assert dataset.get_n_classes() == 2
    assert dataset[0]["target"] == 0


def test_patch_dataset_keeps_features_on_cpu_until_batch_transfer(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.csv"
    feature = tmp_path / "feature.pt"
    torch.save(torch.ones(1, 2), feature)
    pd.DataFrame(
        [
            {
                "case_id": "p1",
                "slide_id": "s1",
                "cancer_type": "A",
                "feature_path": feature,
            }
        ]
    ).to_csv(manifest, index=False)

    sample = ImbalanceDataset(manifest)[0]

    assert sample["features"].device.type == "cpu"

def test_getitems_matches_stacked_getitem_for_arbitrary_indices(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.csv"
    shared = tmp_path / "shared.pt"
    torch.save(torch.arange(12, dtype=torch.float32).reshape(3, 4), shared)
    solo = tmp_path / "solo.pt"
    torch.save(torch.full((1, 4), 9.0), solo)
    rows = [
        {
            "case_id": "p0",
            "slide_id": "s0",
            "cancer_type": "A",
            "feature_path": shared,
            "feature_index": 0,
        },
        {
            "case_id": "p0",
            "slide_id": "s0",
            "cancer_type": "A",
            "feature_path": shared,
            "feature_index": 1,
        },
        {
            "case_id": "p1",
            "slide_id": "s1",
            "cancer_type": "B",
            "feature_path": shared,
            "feature_index": 2,
        },
        {
            "case_id": "p2",
            "slide_id": "s2",
            "cancer_type": "B",
            "feature_path": solo,
        },
    ]
    pd.DataFrame(rows).to_csv(manifest, index=False)
    dataset = ImbalanceDataset(manifest)
    indices = [3, 0, 2, 1, 0]

    batched = dataset.__getitems__(indices)

    stacked_features = torch.stack([dataset[i]["features"] for i in indices])
    stacked_targets = torch.tensor(
        [dataset[i]["target"] for i in indices], dtype=torch.long
    )
    torch.testing.assert_close(batched["features"], stacked_features)
    torch.testing.assert_close(batched["target"], stacked_targets)

def test_bag_dataset_rejects_multiple_labels_for_one_slide(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {
                "case_id": "p1",
                "slide_id": "s1",
                "cancer_type": "N",
                "feature_path": "unused.pt",
            },
            {
                "case_id": "p1",
                "slide_id": "s1",
                "cancer_type": "IC",
                "feature_path": "unused.pt",
            },
        ]
    ).to_csv(manifest, index=False)

    with pytest.raises(ValueError, match="exactly one class"):
        BagFeatureDataset(manifest)

def test_mil_available_support_counts_unique_slides_not_feature_chunks() -> None:
    """MIL availability is the number of slides, not one row per feature chunk."""
    frame = pd.DataFrame(
        [
            {
                "case_id": f"{name}_{slide}",
                "slide_id": f"{name}_{slide}",
                "feature_path": f"{name}_{slide}_{chunk}.pt",
                "cancer_type": name,
            }
            for name in ("A", "B")
            for slide in range(30)
            for chunk in range(2)
        ]
    )

    assert class_support_counts(frame, is_mil=True) == {"A": 30, "B": 30}

def test_bag_dataset_concatenates_every_feature_chunk_of_a_slide(
    tmp_path: Path,
) -> None:
    first, second = tmp_path / "first.pt", tmp_path / "second.pt"
    torch.save(torch.ones(3, 4), first)
    torch.save(torch.full((4, 4), 2.0), second)
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {
                "case_id": "case",
                "slide_id": "slide",
                "cancer_type": "A",
                "feature_path": first,
            },
            {
                "case_id": "case",
                "slide_id": "slide",
                "cancer_type": "A",
                "feature_path": second,
            },
        ]
    ).to_csv(manifest, index=False)

    bag, target = BagFeatureDataset(manifest)[0]

    assert target == 0
    assert len(bag) == 7
    assert bag.sum().item() == pytest.approx(44.0)
