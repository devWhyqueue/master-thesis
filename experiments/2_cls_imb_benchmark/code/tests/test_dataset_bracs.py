from __future__ import annotations

import pandas as pd
from PIL import Image

from imbalance_benchmark.datasets.bracs import (
    assert_patient_disjoint,
    normalize_label,
    split_cases,
)
from imbalance_benchmark.datasets.bracs_tiling import tile_rois


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

    tiled, bag_size = tile_rois(metadata, {"roi_a": image}, tmp_path / "tiles", 256)

    assert bag_size == 2
    assert tiled["patch_id"].tolist() == ["roi_a__000_0_0", "roi_a__001_256_0"]
    assert all(path.endswith(".jpg") for path in tiled["image_path"])


def test_bracs_tiling_caps_each_wsi_at_median(tmp_path) -> None:
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

    tiled, bag_size = tile_rois(metadata, images, tmp_path / "tiles", 256)

    assert bag_size == 3  # median of available tiles per WSI: [2, 3, 5]
    per_wsi = tiled.groupby("slide_id")["patch_id"].count()
    assert per_wsi.max() <= bag_size
    assert per_wsi.to_dict() == {"s1": 2, "s2": 3, "s3": 3}
