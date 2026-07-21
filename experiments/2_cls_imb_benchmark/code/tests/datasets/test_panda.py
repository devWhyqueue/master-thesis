from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from PIL import Image
from imbalance_benchmark.common import compute_sha256
from imbalance_benchmark.datasets.panda import (
    assert_slide_disjoint,
    cell_label,
    isup_label,
    select_subset,
    split_cases,
)
from imbalance_benchmark.datasets.panda_audit import (
    validate_selection,
    validate_tile_inventory,
)
import imbalance_benchmark.datasets as dataset_adapters

def test_panda_wsi_rows_retain_every_eligible_patch() -> None:
    """Uncapped PANDA WSI bags retain the complete tissue-patch inventory."""
    row = pd.Series(
        {
            "slide_id": "slide",
            "slide_label": "ISUP2",
            "provider": "radboud",
            "has_mask": True,
        }
    )
    tiles = pd.DataFrame(
        {
            "patch_id": ["p0", "p1", "p2"],
            "patch_label": ["benign", "cancer", "benign"],
            "image_path": ["p0.jpg", "p1.jpg", "p2.jpg"],
        }
    )

    result = dataset_adapters._panda_slide_rows(row, tiles)

    assert result["patch_id"].tolist() == ["p0", "p1", "p2"]

def test_panda_tile_inventory_rejects_duplicate_selection_rows(
    tmp_path: Path,
) -> None:
    image_a, image_b = tmp_path / "A.jpg", tmp_path / "B.jpg"
    image_a.write_bytes(b"A")
    image_b.write_bytes(b"B")
    selection = pd.DataFrame(
        {
            "slide_id": ["A", "A", "B"],
            "eligible_tile_count": [1, 1, 1],
            "source_level": [0, 0, 0],
            "tile_size": [256, 256, 256],
            "tissue_fraction_min": [0.35, 0.35, 0.35],
            "tissue_intensity_threshold": [210, 210, 210],
        }
    )
    inventory = {
        slide: pd.DataFrame(
            {
                "patch_id": ["p0"],
                "image_path": [str(path)],
                "level": [0],
                "tile_size": [256],
                "x": [0],
                "y": [0],
                "tissue_fraction": [0.5],
                "tissue_intensity_threshold": [210],
            }
        )
        for slide, path in (("A", image_a), ("B", image_b))
    }

    with pytest.raises(ValueError, match="duplicate"):
        validate_tile_inventory(
            selection,
            inventory,
            pd.DataFrame({"slide_id": ["A", "B"]}),
            expected_slides=2,
        )

def test_panda_selection_matches_official_ids_labels_providers_and_masks() -> None:
    official = pd.DataFrame(
        {
            "slide_id": ["A", "B"],
            "slide_label": ["ISUP0", "ISUP1"],
            "provider": ["radboud", "karolinska"],
            "has_mask": [True, False],
        }
    )
    selection = pd.DataFrame(
        {
            "slide_id": ["A", "B"],
            "slide_label": ["ISUP0", "ISUP5"],
            "provider": ["radboud", "karolinska"],
            "has_mask": [True, False],
            "eligible_tile_count": [1, 1],
            "source_level": [0, 0],
            "tile_size": [256, 256],
            "tissue_fraction_min": [0.35, 0.35],
            "tissue_intensity_threshold": [210, 210],
        }
    )

    with pytest.raises(ValueError, match="official slide_label"):
        validate_selection(selection, official, 2)

    selection.loc[1, "slide_label"] = "ISUP1"
    selection.loc[1, "slide_id"] = "C"
    with pytest.raises(ValueError, match="slide IDs"):
        validate_selection(selection, official, 2)

def test_panda_tile_audit_checks_hash_and_provider_specific_mask_label(
    tmp_path: Path,
) -> None:
    image_path, mask_path = tmp_path / "tile.jpg", tmp_path / "mask.tiff"
    image_path.write_bytes(b"tile")
    Image.fromarray(np.full((256, 256), 2, dtype=np.uint8)).save(mask_path)
    selection = pd.DataFrame(
        {
            "slide_id": ["A"],
            "slide_label": ["ISUP0"],
            "provider": ["radboud"],
            "has_mask": [True],
            "eligible_tile_count": [1],
            "source_level": [0],
            "tile_size": [256],
            "tissue_fraction_min": [0.35],
            "tissue_intensity_threshold": [210],
        }
    )
    tiles = pd.DataFrame(
        {
            "patch_id": ["p0"],
            "patch_label": ["cancer"],
            "image_path": [str(image_path)],
            "sha256": [compute_sha256(image_path)],
            "level": [0],
            "tile_size": [256],
            "x": [0],
            "y": [0],
            "tissue_fraction": [0.5],
            "tissue_intensity_threshold": [210],
        }
    )
    official = pd.DataFrame(
        {
            "slide_id": ["A"],
            "provider": ["radboud"],
            "has_mask": [True],
            "mask_path": [str(mask_path)],
        }
    )

    with pytest.raises(ValueError, match="patch label"):
        validate_tile_inventory(selection, {"A": tiles}, official, 1)

    tiles.loc[0, "patch_label"] = "benign"
    tiles.loc[0, "sha256"] = "wrong"
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_tile_inventory(selection, {"A": tiles}, official, 1)

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

def test_panda_split_cases_are_slide_disjoint() -> None:
    frame = pd.DataFrame(
        {
            "case_id": [f"slide{i}" for i in range(18)],
            "slide_label": [isup_label(i % 6) for i in range(18)],
        }
    )
    assigned = frame.merge(split_cases(frame, 0), on="case_id", how="inner")

    assert_slide_disjoint(assigned)
    assert set(assigned["split"]).issubset({"train", "validation", "test"})
