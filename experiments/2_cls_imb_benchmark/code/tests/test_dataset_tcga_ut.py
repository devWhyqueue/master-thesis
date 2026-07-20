from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from imbalance_benchmark.common import compute_sha256
from imbalance_benchmark.datasets import build_manifest
from imbalance_benchmark.datasets.feature_provenance import resolve_feature_provenance
from imbalance_benchmark.datasets.tcga_ut import (
    assert_case_disjoint,
    assign_class_splits,
    build_feature_manifest,
    collect_slide_labels,
    split_cases,
    strip_feature_suffix,
    tcga_case_id,
)


def _tcga_config(
    raw_root: Path,
    feature_dir: Path,
    *,
    expected_slides: int,
    expected_classes: int,
    expected_patches: int,
) -> dict[str, object]:
    manifest_path = feature_dir / "feature_provenance_manifest.json"
    chunks = {}
    for path in sorted(feature_dir.glob("*.pt")):
        tensor = torch.load(path, weights_only=False)
        chunks[path.name] = {
            "tensor_sha256": compute_sha256(path),
            "row_count": len(tensor),
            "feature_dim": tensor.shape[1],
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "patch_order_sha256": "0" * 64,
        }
    manifest_path.write_text(
        json.dumps(
            {
                "provenance": resolve_feature_provenance({"dtype": "float16"}),
                "chunks": chunks,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    manifest_path.with_suffix(".json.sha256").write_text(
        compute_sha256(manifest_path), encoding="utf-8"
    )
    return {
        "dataset": {
            "name": "tcga_ut",
            "regime": "patch",
            "raw_root": str(raw_root),
            "feature_dir": str(feature_dir),
            "feature_provenance_manifest": str(manifest_path),
            "expected_slide_count": expected_slides,
            "expected_class_count": expected_classes,
            "expected_patch_count": expected_patches,
        },
        "feature_extraction": {"dtype": "float16"},
    }


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


def test_tcga_manifest_rejects_raw_slides_without_features(tmp_path: Path) -> None:
    raw_root, feature_dir = tmp_path / "raw", tmp_path / "features"
    (raw_root / "LUAD" / "train" / "TCGA-AB-0001-01Z").mkdir(parents=True)
    (raw_root / "LUAD" / "train" / "TCGA-AB-0002-01Z").mkdir(parents=True)
    feature_dir.mkdir()
    torch.save(torch.ones(2, 2560), feature_dir / "TCGA-AB-0001-01Z_0.pt")
    config = _tcga_config(
        raw_root,
        feature_dir,
        expected_slides=2,
        expected_classes=1,
        expected_patches=2,
    )

    with pytest.raises(ValueError, match="raw slides without features"):
        build_manifest(config)


def test_tcga_manifest_rejects_label_conflicts(tmp_path: Path) -> None:
    raw_root, feature_dir = tmp_path / "raw", tmp_path / "features"
    slide = "TCGA-AB-0001-01Z"
    (raw_root / "LUAD" / "train" / slide).mkdir(parents=True)
    (raw_root / "LUSC" / "train" / slide).mkdir(parents=True)
    feature_dir.mkdir()
    torch.save(torch.ones(2, 2560), feature_dir / f"{slide}_0.pt")
    config = _tcga_config(
        raw_root,
        feature_dir,
        expected_slides=1,
        expected_classes=1,
        expected_patches=2,
    )

    with pytest.raises(ValueError, match="label conflicts"):
        build_manifest(config)


def test_tcga_manifest_rejects_unmatched_feature_chunks(tmp_path: Path) -> None:
    raw_root, feature_dir = tmp_path / "raw", tmp_path / "features"
    slide = "TCGA-AB-0001-01Z"
    (raw_root / "LUAD" / "train" / slide).mkdir(parents=True)
    feature_dir.mkdir()
    torch.save(torch.ones(2, 2560), feature_dir / f"{slide}_0.pt")
    torch.save(torch.ones(2, 2560), feature_dir / "unknown_0.pt")
    config = _tcga_config(
        raw_root,
        feature_dir,
        expected_slides=1,
        expected_classes=1,
        expected_patches=2,
    )

    with pytest.raises(ValueError, match="unmatched feature chunks"):
        build_manifest(config)


def test_tcga_manifest_rejects_wrong_patch_total(tmp_path: Path) -> None:
    raw_root, feature_dir = tmp_path / "raw", tmp_path / "features"
    slide = "TCGA-AB-0001-01Z"
    (raw_root / "LUAD" / "train" / slide).mkdir(parents=True)
    feature_dir.mkdir()
    torch.save(torch.ones(2, 2560), feature_dir / f"{slide}_0.pt")
    config = _tcga_config(
        raw_root,
        feature_dir,
        expected_slides=1,
        expected_classes=1,
        expected_patches=3,
    )

    with pytest.raises(ValueError, match="patch count"):
        build_manifest(config)


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    [
        ("tensor_sha256", "0" * 64, "tensor hash"),
        ("feature_dim", 42, "feature dimension"),
        ("dtype", "float64", "dtype"),
        ("patch_order_sha256", "missing", "patch order"),
    ],
)
def test_tcga_manifest_rejects_invalid_tensor_provenance(
    tmp_path: Path, field: str, bad_value: object, message: str
) -> None:
    raw_root, feature_dir = tmp_path / "raw", tmp_path / "features"
    slide = "TCGA-AB-0001-01Z"
    (raw_root / "LUAD" / "train" / slide).mkdir(parents=True)
    feature_dir.mkdir()
    tensor_path = feature_dir / f"{slide}_0.pt"
    torch.save(torch.ones(2, 2560), tensor_path)
    config = _tcga_config(
        raw_root,
        feature_dir,
        expected_slides=1,
        expected_classes=1,
        expected_patches=2,
    )
    manifest_path = Path(config["dataset"]["feature_provenance_manifest"])
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["chunks"][tensor_path.name][field] = bad_value
    manifest_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    manifest_path.with_suffix(".json.sha256").write_text(
        compute_sha256(manifest_path), encoding="utf-8"
    )

    with pytest.raises(ValueError, match=message):
        build_manifest(config)


def test_tcga_manifest_requires_signed_pinned_provenance(tmp_path: Path) -> None:
    raw_root, feature_dir = tmp_path / "raw", tmp_path / "features"
    slide = "TCGA-AB-0001-01Z"
    (raw_root / "LUAD" / "train" / slide).mkdir(parents=True)
    feature_dir.mkdir()
    torch.save(torch.ones(2, 2560), feature_dir / f"{slide}_0.pt")
    config = _tcga_config(
        raw_root,
        feature_dir,
        expected_slides=1,
        expected_classes=1,
        expected_patches=2,
    )
    manifest_path = Path(config["dataset"]["feature_provenance_manifest"])
    manifest_path.with_suffix(".json.sha256").unlink()

    with pytest.raises(ValueError, match="signed provenance"):
        build_manifest(config)


def test_tcga_manifest_rejects_wrong_encoder_provenance(tmp_path: Path) -> None:
    raw_root, feature_dir = tmp_path / "raw", tmp_path / "features"
    slide = "TCGA-AB-0001-01Z"
    (raw_root / "LUAD" / "train" / slide).mkdir(parents=True)
    feature_dir.mkdir()
    torch.save(torch.ones(2, 2560), feature_dir / f"{slide}_0.pt")
    config = _tcga_config(
        raw_root,
        feature_dir,
        expected_slides=1,
        expected_classes=1,
        expected_patches=2,
    )
    manifest_path = Path(config["dataset"]["feature_provenance_manifest"])
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["provenance"]["encoder_revision"] = "wrong"
    manifest_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    manifest_path.with_suffix(".json.sha256").write_text(
        compute_sha256(manifest_path), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="pinned Virchow2 provenance"):
        build_manifest(config)


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
    assigned = slide_manifest.merge(
        split_cases(slide_manifest, 0), on="case_id", how="inner"
    )

    assert_case_disjoint(assigned)
    assert set(assigned["split"]).issubset({"train", "validation", "test"})
