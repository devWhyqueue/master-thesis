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


def test_check_grid_eligibility_fallback():
    """Infeasible ladder triggers depth fallback then breadth fallback."""
    # Build df where each class has 10 patients with 16 patches, but none with 32
    rows = []
    for c in ["c1", "c2"]:
        for p in range(10):
            for patch in range(16):
                rows.append(
                    {
                        "cancer_type": c,
                        "case_id": f"p_{c}_{p}",
                        "slide_id": f"s_{c}_{p}",
                        "patch_id": f"pt_{c}_{p}_{patch}",
                    }
                )
    df = pd.DataFrame(rows)
    train_dfs = {0: df, 1: df, 2: df}

    # Initial ladder (5, 10, 20) x (8, 16, 32): 32 fails and max_g=20 fails
    # Fallback depth: (4, 8, 16), max_g still 20 (fails)
    # Fallback breadth: (4, 8, 16) with max_g=16 (fails, only 10 patients)
    with pytest.raises(RuntimeError, match="infeasible"):
        check_grid_eligibility(train_dfs, ["c1", "c2"])

    # If 25 patients each have 16 patches:
    rows25 = []
    for c in ["c1", "c2"]:
        for p in range(25):
            for patch in range(16):
                rows25.append(
                    {
                        "cancer_type": c,
                        "case_id": f"p_{c}_{p}",
                        "slide_id": f"s_{c}_{p}",
                        "patch_id": f"pt_{c}_{p}_{patch}",
                    }
                )
    df25 = pd.DataFrame(rows25)
    train_dfs25 = {0: df25, 1: df25, 2: df25}

    b_lad, d_lad, audit = check_grid_eligibility(train_dfs25, ["c1", "c2"])
    # 25 >= 20, but depth 32 is missing -> falls back to depth ladder (4, 8, 16)
    assert b_lad == (5, 10, 20)
    assert d_lad == (4, 8, 16)
    assert "0" in audit


def test_sample_cell_draw_exact_budget():
    """sample_cell_draw selects G patients x m patches per class."""
    rows = []
    for c in ["class_0", "class_1"]:
        for p in range(10):
            for s in range(2):
                for patch in range(10):
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

