from __future__ import annotations

import numpy as np
import pandas as pd

from imbalance_benchmark.datasets.panda import (
    assert_slide_disjoint,
    cell_label,
    isup_label,
    select_subset,
    split_cases,
)


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
