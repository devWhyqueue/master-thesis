import json

import pandas as pd
from PIL import Image

from analysis.evaluation.native_tuning_aggregate import select_winners
from data.bracs.prepare import (
    assert_patient_disjoint,
    normalize_label,
    split_cases,
    tile_rois,
)


def test_bracs_label_normalization_maps_seven_subtypes() -> None:
    assert normalize_label("Normal") == "N"
    assert normalize_label("Pathological Benign") == "PB"
    assert normalize_label("Usual Ductal Hyperplasia") == "UDH"
    assert normalize_label("Flat Epithelial Atypia") == "FEA"
    assert normalize_label("Atypical Ductal Hyperplasia") == "ADH"
    assert normalize_label("Ductal Carcinoma in Situ") == "DCIS"
    assert normalize_label("Invasive Carcinoma") == "IC"


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

    tiled = tile_rois(metadata, {"roi_a": image}, tmp_path / "tiles", 256, 30)

    assert tiled["patch_id"].tolist() == ["roi_a__000_0_0", "roi_a__001_256_0"]
    assert all(path.endswith(".jpg") for path in tiled["image_path"])


def test_native_tuning_aggregate_selects_winner() -> None:
    frame = pd.DataFrame(
        [
            {
                "benchmark": "patch",
                "method": "patch_feature_weighted_ce",
                "variant": "a",
                "params": json.dumps({"weight_power": 1.0}),
                "seed": seed,
                "val_macro_f1": 0.5,
                "val_balanced_accuracy": 0.5,
                "test_macro_f1": 0.4,
                "test_balanced_accuracy": 0.4,
            }
            for seed in (0, 1, 2)
        ]
        + [
            {
                "benchmark": "patch",
                "method": "patch_feature_weighted_ce",
                "variant": "b",
                "params": json.dumps({"weight_power": 2.0}),
                "seed": seed,
                "val_macro_f1": 0.6,
                "val_balanced_accuracy": 0.5,
                "test_macro_f1": 0.7,
                "test_balanced_accuracy": 0.7,
            }
            for seed in (0, 1, 2)
        ]
    )

    selected = select_winners(frame, allow_incomplete=False)

    assert selected[0]["variant"] == "b"
    assert selected[0]["regime"] == "native"
