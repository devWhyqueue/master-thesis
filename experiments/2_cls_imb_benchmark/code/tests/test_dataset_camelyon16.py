from __future__ import annotations

import numpy as np
import pandas as pd

from imbalance_benchmark.datasets.camelyon16 import (
    CELL,
    assert_slide_disjoint,
    patch_labels,
    split_cases,
)


def test_camelyon16_patch_labels_decode_column_major_cells() -> None:
    mask = np.zeros((CELL * 2, CELL * 2), dtype=np.uint8)
    mask[0:CELL, 0:CELL] = 2  # row=0, col=0 -> patch_id 0 (n_rows=2: col,row=divmod(0,2)=(0,0))
    mask[CELL:, CELL:] = 2  # row=1, col=1 -> patch_id = col*n_rows + row = 1*2+1 = 3

    labels = patch_labels(mask, [0, 1, 2, 3])

    assert labels[0] == "tumor"
    assert labels[3] == "tumor"
    assert labels[1] == "normal"
    assert labels[2] == "normal"


def test_camelyon16_patch_labels_out_of_bounds_is_normal() -> None:
    mask = np.zeros((CELL, CELL), dtype=np.uint8)
    assert patch_labels(mask, [50]) == ["normal"]


def test_camelyon16_split_cases_are_slide_disjoint() -> None:
    frame = pd.DataFrame(
        {
            "case_id": [f"slide_{i}" for i in range(20)],
            "slide_label": ["tumor" if i < 10 else "normal" for i in range(20)],
        }
    )
    assigned = frame.merge(split_cases(frame, 0), on="case_id", how="inner")

    assert_slide_disjoint(assigned)
    assert set(assigned["split"]).issubset({"train", "validation", "test"})


def test_camelyon16_slide_disjoint_validation_rejects_leakage() -> None:
    frame = pd.DataFrame({"case_id": ["s1", "s1"], "split": ["train", "test"]})
    try:
        assert_slide_disjoint(frame)
    except ValueError as error:
        assert "slide leakage" in str(error)
    else:
        raise AssertionError("Expected slide leakage to raise.")
