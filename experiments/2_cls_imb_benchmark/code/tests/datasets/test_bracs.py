from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from PIL import Image
from imbalance_benchmark.commands.pilot import _pilot_setup
from imbalance_benchmark.datasets.bracs import (
    assert_patient_disjoint,
    list_slide_tiles,
    load_roi_metadata,
    load_wsi_metadata,
    normalize_label,
    split_cases,
    tile_rois,
)
from imbalance_benchmark.datasets.bracs import LABELS as BRACS_LABELS
from imbalance_benchmark.datasets.bracs import wsi as bracs_wsi
import imbalance_benchmark.datasets as dataset_adapters

def test_bracs_wsi_uses_official_slide_metadata_without_roi_supervision(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """BRACS WSI bags use official slide labels and never inspect ROI metadata."""
    tile = tmp_path / "tile.jpg"
    tile.touch()
    pd.DataFrame(
        {
            "slide_id": ["BRACS_1"],
            "image_path": [str(tile)],
            "magnification": ["20x"],
            "tile_size": [256],
            "x": [0],
            "y": [0],
            "otsu_foreground_fraction": [0.5],
            "grayscale_std": [10.0],
            "canny_edge_count": [1],
            "tissue_neighbors": [2],
            "sha256": [
                "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
            ],
        }
    ).to_csv(tmp_path / "tile_manifest.csv", index=False)
    monkeypatch.setattr(
        dataset_adapters.bracs,
        "load_roi_metadata",
        lambda *_: (_ for _ in ()).throw(AssertionError("WSI must not read ROIs")),
    )
    monkeypatch.setattr(
        dataset_adapters.bracs.wsi,
        "load_wsi_metadata",
        lambda *_: pd.DataFrame(
            {
                "dataset": ["bracs"],
                "case_id": ["patient_1"],
                "slide_id": ["BRACS_1"],
                "slide_label": ["DCIS"],
            }
        ),
    )
    rows = dataset_adapters._build_bracs(
        {
            "dataset": {
                "root": str(tmp_path),
                "wsi_tile_root": str(tmp_path),
                "expected_wsi_count": 1,
                "regime": "wsi",
            }
        }
    )

    assert rows["cancer_type"].tolist() == ["DCIS"]
    assert rows["slide_label"].tolist() == ["DCIS"]
    assert "patch_label" not in rows

def test_bracs_wsi_manifest_rejects_different_slide_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    official = pd.DataFrame(
        {
            "case_id": ["pA", "pB"],
            "slide_id": ["A", "B"],
            "slide_label": ["N", "N"],
        }
    )
    tiles = pd.DataFrame(
        {"slide_id": ["A", "C"], "image_path": ["A.jpg", "C.jpg"]}
    )
    monkeypatch.setattr(bracs_wsi, "load_wsi_metadata", lambda *_args: official)
    monkeypatch.setattr(bracs_wsi, "load_tile_manifest", lambda *_args: tiles)
    monkeypatch.setattr(
        bracs_wsi.metadata,
        "split_cases",
        lambda *_args: pd.DataFrame({"case_id": ["pA"], "split": ["train"]}),
    )

    with pytest.raises(ValueError, match="slide IDs"):
        bracs_wsi.build_manifest(tmp_path, tmp_path, 0, expected_slides=2)

def test_bracs_roi_metadata_joins_release_filenames_to_patients(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    (tmp_path / "BRACS.xlsx").touch()
    roi_dir = tmp_path / "BRACS_RoI" / "latest_version" / "train" / "0_N"
    roi_dir.mkdir(parents=True)
    (roi_dir / "BRACS_1_N_2.png").touch()
    monkeypatch.setattr(
        pd,
        "read_excel",
        lambda *_args, **_kwargs: {
            "WSI_Information": pd.DataFrame(
                {
                    "WSI Filename": ["BRACS_1"],
                    "Patient Id": [85],
                    "RoI ": [24],
                    "WSI label": ["PB"],
                }
            )
        },
    )

    metadata = load_roi_metadata(tmp_path)

    assert metadata[
        ["case_id", "slide_id", "roi_id", "cancer_type", "lesion_type"]
    ].to_dict("records") == [
        {
            "case_id": "85",
            "slide_id": "BRACS_1",
            "roi_id": "BRACS_1_N_2",
            "cancer_type": "N",
            "lesion_type": "benign",
        }
    ]

def test_bracs_label_normalization_maps_seven_subtypes() -> None:
    assert normalize_label("Normal") == "N"
    assert normalize_label("Pathological Benign") == "PB"
    assert normalize_label("Usual Ductal Hyperplasia") == "UDH"
    assert normalize_label("Flat Epithelial Atypia") == "FEA"
    assert normalize_label("Atypical Ductal Hyperplasia") == "ADH"
    assert normalize_label("Ductal Carcinoma in Situ") == "DCIS"
    assert normalize_label("Invasive Carcinoma") == "IC"
    assert normalize_label("not a subtype") is None

def test_bracs_split_cases_are_patient_disjoint() -> None:
    frame = pd.DataFrame(
        [
            {"case_id": f"p{i}", "slide_id": f"s{i}", "cancer_type": label}
            for i, label in enumerate(["N", "N", "PB", "PB", "IC", "IC", "ADH"])
        ]
    )
    assigned = frame.merge(split_cases(frame, 0), on="case_id", how="inner")

    assert_patient_disjoint(assigned)
    assert set(assigned["split"]).issubset({"train", "validation", "test"})

def test_bracs_patient_disjoint_validation_rejects_leakage() -> None:
    frame = pd.DataFrame({"case_id": ["p1", "p1"], "split": ["train", "test"]})
    try:
        assert_patient_disjoint(frame)
    except ValueError as error:
        assert "patient leakage" in str(error)
    else:
        raise AssertionError("Expected patient leakage to raise.")

def test_bracs_roi_tiling_is_deterministic(tmp_path) -> None:
    image = tmp_path / "roi_a.jpg"
    Image.new("RGB", (512, 256), color=(120, 80, 40)).save(image)
    metadata = pd.DataFrame(
        [
            {
                "case_id": "p1",
                "slide_id": "s1",
                "roi_id": "roi_a",
                "cancer_type": "N",
                "lesion_type": "benign",
            }
        ]
    )

    tiled = tile_rois(metadata, {"roi_a": image}, tmp_path / "tiles", 256)

    assert tiled["patch_id"].tolist() == ["roi_a__000_0_0", "roi_a__001_256_0"]
    assert all(path.endswith(".jpg") for path in tiled["image_path"])

def test_bracs_tiling_retains_every_complete_roi_patch(tmp_path) -> None:
    images = {}
    rows = []
    for slide, n_tiles in (("s1", 2), ("s2", 3), ("s3", 5)):
        roi_id = f"roi_{slide}"
        path = tmp_path / f"{roi_id}.jpg"
        Image.new("RGB", (256 * n_tiles, 256), color=(120, 80, 40)).save(path)
        images[roi_id] = path
        rows.append(
            {
                "case_id": slide,
                "slide_id": slide,
                "roi_id": roi_id,
                "cancer_type": "N",
                "lesion_type": "benign",
            }
        )
    metadata = pd.DataFrame(rows)

    tiled = tile_rois(metadata, images, tmp_path / "tiles", 256)

    per_wsi = tiled.groupby("slide_id")["patch_id"].count()
    assert per_wsi.to_dict() == {"s1": 2, "s2": 3, "s3": 5}

def test_bracs_wsi_metadata_uses_official_labels_without_roi_derivation(
    tmp_path,
) -> None:
    metadata_path = tmp_path / "wsi_metadata.csv"
    pd.DataFrame(
        {
            "Patient ID": ["p1", "p2"],
            "WSI Filename": ["BRACS_1.svs", "BRACS_2.svs"],
            "WSI Label": ["ADH", "Invasive Carcinoma"],
            "Number of RoIs": [3, 0],
        }
    ).to_csv(metadata_path, index=False)

    metadata = load_wsi_metadata(tmp_path, metadata_path)

    assert metadata[["slide_id", "case_id", "slide_label"]].to_dict("records") == [
        {"slide_id": "BRACS_1", "case_id": "p1", "slide_label": "ADH"},
        {"slide_id": "BRACS_2", "case_id": "p2", "slide_label": "IC"},
    ]

def test_bracs_wsi_tiles_have_deterministic_order(tmp_path) -> None:
    slide_dir = tmp_path / "BRACS_1"
    slide_dir.mkdir()
    for name in ("patch_10.jpg", "patch_02.jpg", "patch_01.jpg"):
        Image.new("RGB", (256, 256)).save(slide_dir / name)

    tiles = list_slide_tiles(tmp_path, "BRACS_1")

    assert [path.name for path in tiles] == [
        "patch_01.jpg",
        "patch_02.jpg",
        "patch_10.jpg",
    ]

def test_pilot_uses_the_same_semantic_bracs_order_as_the_freeze(tmp_path: Path) -> None:
    rows = [
        {
            "case_id": f"patient-{name}",
            "slide_id": f"slide-{name}",
            "cancer_type": name,
            "split": "train",
        }
        for name in BRACS_LABELS
    ]
    pd.DataFrame(rows).to_csv(tmp_path / "manifest.csv", index=False)

    _, class_names, *_ = _pilot_setup({"data": tmp_path}, {"dataset": {}})

    assert class_names == list(BRACS_LABELS)
