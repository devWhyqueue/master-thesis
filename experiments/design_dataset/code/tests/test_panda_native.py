import json
import sys

import numpy as np
import pandas as pd

from analysis.plotting.panda_native_report import write_result_table
from data.panda import prepare
from data.panda.masks import cell_label
from data.panda.metadata import isup_label, select_subset


def test_panda_mask_decode_is_provider_specific() -> None:
    radboud = np.array([[3, 4], [4, 5]])
    assert cell_label(radboud, "radboud") == "cancer"
    assert cell_label(np.array([[1, 2], [2, 1]]), "radboud") == "benign"
    # Radboud Gleason values are benign under the Karolinska value map.
    assert cell_label(radboud, "karolinska") == "benign"
    assert cell_label(np.array([[2, 2], [2, 0]]), "karolinska") == "cancer"


def test_panda_isup_label_and_subset_stratification() -> None:
    assert isup_label(0) == "ISUP0"
    assert isup_label(5) == "ISUP5"
    frame = pd.DataFrame(
        {
            "slide_id": [f"s{i}" for i in range(60)],
            "isup_grade": [i % 6 for i in range(60)],
            "provider": ["radboud" if i % 2 else "karolinska" for i in range(60)],
        }
    )
    subset = select_subset(frame, 30, seed=0)
    assert 24 <= len(subset) <= 36
    assert set(subset["isup_grade"]) == set(range(6))


def test_panda_prepare_writes_disjoint_manifests_and_report(tmp_path) -> None:
    root = tmp_path / "work"
    root.mkdir()
    tiles_dir = root / "tiles"
    tiles_dir.mkdir()
    slides = []
    for i in range(18):
        slide_id = f"slide{i}"
        grade = i % 6
        slides.append(
            {
                "slide_id": slide_id,
                "provider": "radboud",
                "isup_grade": grade,
                "slide_label": isup_label(grade),
                "image_path": "x",
                "mask_path": "x",
                "has_mask": True,
            }
        )
        label = "cancer" if grade >= 3 else "benign"
        pd.DataFrame(
            {
                "slide_id": slide_id,
                "patch_id": range(5),
                "image_path": [f"{slide_id}_{j}.jpg" for j in range(5)],
                "patch_label": [label] * 5,
            }
        ).to_csv(tiles_dir / f"{slide_id}.csv", index=False)
    selection_path = root / "selected_slides.csv"
    pd.DataFrame(slides).to_csv(selection_path, index=False)

    argv = [
        "prepare",
        f"--output-root={root}",
        f"--selection-path={selection_path}",
        f"--tiles-dir={tiles_dir}",
        "--seeds",
        "0",
    ]
    old = sys.argv
    sys.argv = argv
    try:
        prepare.main()
    finally:
        sys.argv = old

    wsi = pd.read_csv(root / "manifests" / "native_seed=0" / "manifest_splits.csv")
    assert (wsi.groupby("case_id")["split"].nunique() == 1).all()
    assert set(wsi["cancer_type"]) <= {isup_label(g) for g in range(6)}
    order = json.loads(
        (root / "manifests" / "native_seed=0" / "class_order.json").read_text()
    )
    assert order == ["benign", "cancer"]
    report = json.loads((root / "panda_prepare_report.json").read_text())
    assert set(report["class_counts_wsi"]) == {isup_label(g) for g in range(6)}
    assert report["recommended_benchmark_mode"] == "native"
    assert report["imbalance_ratio_patch"] >= 1.0


def test_panda_patch_result_table_has_progan_placeholder(tmp_path) -> None:
    frame = pd.DataFrame(
        [
            {
                "method": "patch_feature_ce",
                "accuracy": 0.9,
                "balanced_accuracy": 0.9,
                "macro_f1": 0.9,
            }
        ]
    )
    path = tmp_path / "panda_result_summary_patch.tex"
    write_result_table(frame, path, "patch")
    text = path.read_text()
    assert "ProGAN augmentation & -- & -- & --" in text


def test_panda_patch_result_table_drops_placeholder_when_progan_present(tmp_path) -> None:
    frame = pd.DataFrame(
        [
            {"method": m, "accuracy": 0.9, "balanced_accuracy": 0.9, "macro_f1": f}
            for m, f in (("patch_feature_ce", 0.9), ("patch_feature_progan_aug", 0.8))
        ]
    )
    path = tmp_path / "panda_result_summary_patch.tex"
    write_result_table(frame, path, "patch")
    text = path.read_text()
    assert "-- & -- & --" not in text
    assert text.count("ProGAN augmentation") == 1
