"""Unit tests for patient sampling and round-robin patch selection."""

from __future__ import annotations

import pandas as pd
import pytest

from breadth.sampling import (
    check_grid_eligibility,
    derive_draw_seed,
    eligible_patients_by_class,
    sample_cell_draw,
    sample_patient_patches_round_robin,
)


def test_derive_draw_seed_deterministic_and_unique():
    """Seed generation is deterministic and sensitive to all parameters."""
    s1 = derive_draw_seed(0, 0, 5, 8, 0, 0)
    s2 = derive_draw_seed(0, 0, 5, 8, 0, 0)
    s3 = derive_draw_seed(0, 0, 5, 8, 1, 0)
    s4 = derive_draw_seed(0, 0, 10, 8, 0, 0)

    assert s1 == s2
    assert s1 != s3
    assert s1 != s4
    assert 0 <= s1 < 2**31 - 1


def test_eligible_patients_by_class():
    """Only patients meeting or exceeding depth threshold are eligible."""
    data = {
        "cancer_type": ["clsA"] * 6 + ["clsB"] * 4,
        "case_id": ["p1", "p1", "p1", "p2", "p2", "p3"] + ["p4", "p4", "p4", "p4"],
    }
    df = pd.DataFrame(data)

    elig_m2 = eligible_patients_by_class(df, "clsA", m=2)
    assert elig_m2 == ["p1", "p2"]

    elig_m3 = eligible_patients_by_class(df, "clsA", m=3)
    assert elig_m3 == ["p1"]

    elig_m4 = eligible_patients_by_class(df, "clsA", m=4)
    assert elig_m4 == []


def test_sample_patient_patches_round_robin():
    """Round-robin distributes picks across patient slides."""
    patient_df = pd.DataFrame(
        {
            "slide_id": ["s1", "s1", "s1", "s2", "s2", "s3"],
            "patch_id": ["p0", "p1", "p2", "p3", "p4", "p5"],
        }
    )

    # Pick 4 patches: s1->p0, s2->p3, s3->p5, then s1->p1
    selected = sample_patient_patches_round_robin(patient_df, m=4)
    assert len(selected) == 4
    picked_patches = patient_df.loc[selected, "patch_id"].tolist()
    assert picked_patches == ["p0", "p3", "p5", "p1"]


def test_sample_patient_patches_insufficient_raises():
    """Attempting to select more patches than available raises ValueError."""
    patient_df = pd.DataFrame(
        {
            "slide_id": ["s1", "s1"],
            "patch_id": ["p0", "p1"],
        }
    )
    with pytest.raises(ValueError, match="Patient has only 2 patches, need 3"):
        sample_patient_patches_round_robin(patient_df, m=3)


def test_check_grid_eligibility_requires_common_pool():
    """Shallow-only patients cannot rescue an infeasible common pool."""
    frame = pd.DataFrame(
        [
            {"cancer_type": "c", "case_id": f"p{patient}"}
            for patient in range(25)
            for _ in range(32 if patient < 16 else 16)
        ]
    )
    with pytest.raises(RuntimeError, match="infeasible"):
        check_grid_eligibility({0: frame}, ["c"])
    breadth, depth, audit = check_grid_eligibility(
        {0: frame}, ["c"], breadth_ladder=(4, 8, 16)
    )
    assert breadth == (4, 8, 16)
    assert depth == (8, 16, 32)
    assert set(audit["0"]["c"].values()) == {16}


def test_depth_draws_share_patients_and_nested_patches():
    """All depths use maximum-depth patients and prefixes of their patches."""
    frame = pd.DataFrame(
        [
            {
                "cancer_type": "c",
                "case_id": f"p{patient}",
                "slide_id": f"s{patch % 3}",
                "patch_id": f"p{patient}_{patch:02d}",
            }
            for patient in range(30)
            for patch in range(32 if patient < 20 else 16)
        ]
    )
    for draw in range(5):
        cells = [sample_cell_draw(frame, ["c"], 5, m, 0, draw) for m in (8, 16, 32)]
        assert set(cells[0].case_id) == set(cells[1].case_id) == set(cells[2].case_id)
        assert set(cells[0].case_id) <= {f"p{i}" for i in range(20)}
        assert set(cells[0].patch_id) < set(cells[1].patch_id) < set(cells[2].patch_id)
    with pytest.raises(RuntimeError, match="eligible"):
        sample_cell_draw(frame, ["c"], 21, 8, 0, 0)
    with pytest.raises(ValueError, match="Depth"):
        sample_cell_draw(frame, ["c"], 5, 64, 0, 0)


def test_sample_cell_draw_exact_budget():
    """sample_cell_draw selects G patients x m patches per class."""
    rows = []
    for c in ["class_0", "class_1"]:
        for p in range(10):
            for s in range(2):
                for patch in range(16):
                    rows.append(
                        {
                            "cancer_type": c,
                            "case_id": f"case_{c}_{p}",
                            "slide_id": f"slide_{c}_{p}_{s}",
                            "patch_id": f"pt_{c}_{p}_{s}_{patch}",
                            "feature_path": "feat.pt",
                            "feature_index": 0,
                        }
                    )
    train_df = pd.DataFrame(rows)

    g, m = 5, 8
    sample_df = sample_cell_draw(
        train_df, ["class_0", "class_1"], g=g, m=m, split_index=0, draw_index=0
    )

    assert len(sample_df) == 2 * g * m
    for c in ["class_0", "class_1"]:
        c_df = sample_df[sample_df["cancer_type"] == c]
        assert len(c_df) == g * m
        assert c_df["case_id"].nunique() == g
        counts = c_df["case_id"].value_counts()
        assert all(cnt == m for cnt in counts)
