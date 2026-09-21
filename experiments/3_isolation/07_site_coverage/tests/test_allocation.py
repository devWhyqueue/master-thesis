"""Unit tests for site pools and the three nested site-coverage allocations."""

from __future__ import annotations

import pandas as pd
import pytest

from breadth.sampling import sample_cell_draw

from sites import ALLOCATIONS, BACKGROUND_CELL, N_CORE_SITES, N_SITES, PATIENTS_PER_SITE
from sites.allocation import draw_allocations, site_class_names, site_pool

N_PATCHES = 32  # >= max(DEPTH_LADDER), so every patient is grid-eligible


def _patient_rows(cancer_type: str, case_id: str, n_patches: int = N_PATCHES) -> list[dict]:
    return [
        {
            "cancer_type": cancer_type,
            "case_id": case_id,
            "slide_id": f"{case_id}_s0",
            "patch_id": f"{case_id}_p{patch:03d}",
        }
        for patch in range(n_patches)
    ]


def _site_class_frame(cancer_type: str, n_sites: int = 12, patients_per_site: int = 5) -> pd.DataFrame:
    """A class whose site pool exceeds N_SITES, each site above the pool threshold."""
    rows: list[dict] = []
    for site in range(n_sites):
        for patient in range(patients_per_site):
            case_id = f"TCGA-{site:02d}-{patient:04d}"
            rows.extend(_patient_rows(cancer_type, case_id))
    return pd.DataFrame(rows)


def _background_frame(cancer_type: str, n_patients: int = 10) -> pd.DataFrame:
    """A class whose patients all share one site, so it never enters the site contrast."""
    rows: list[dict] = []
    for patient in range(n_patients):
        case_id = f"TCGA-00-{patient:04d}"
        rows.extend(_patient_rows(cancer_type, case_id))
    return pd.DataFrame(rows)


@pytest.fixture
def frame() -> pd.DataFrame:
    return pd.concat(
        [_site_class_frame("site_cls"), _background_frame("bg_cls")], ignore_index=True
    )


def test_site_pool_requires_minimum_patients_per_site():
    """Sites with fewer than PATIENTS_PER_SITE eligible patients are dropped."""
    rows = []
    for patient in range(3):  # below PATIENTS_PER_SITE (4)
        rows.extend(_patient_rows("c", f"TCGA-01-{patient:04d}"))
    for patient in range(PATIENTS_PER_SITE):
        rows.extend(_patient_rows("c", f"TCGA-02-{patient:04d}"))
    frame = pd.DataFrame(rows)

    pool = site_pool(frame, "c")
    assert set(pool) == {"02"}
    assert len(pool["02"]) == PATIENTS_PER_SITE


def test_site_class_names_requires_pool_in_every_split(frame: pd.DataFrame):
    """A class must clear the site-pool threshold in every split to qualify."""
    thin_split = frame[frame["cancer_type"] == "bg_cls"]  # site_cls missing entirely
    train_dfs = {0: frame, 1: frame, 2: thin_split}
    assert site_class_names(train_dfs, ["site_cls", "bg_cls"]) == []

    train_dfs_ok = {0: frame, 1: frame, 2: frame}
    assert site_class_names(train_dfs_ok, ["site_cls", "bg_cls"]) == ["site_cls"]


def test_draw_allocations_site_counts(frame: pd.DataFrame):
    """Ten sites are drawn, split into five core and five added."""
    _, site_record = draw_allocations(frame, ["site_cls", "bg_cls"], ["site_cls"], 0, 0)
    core = site_record["site_cls"]["core"]
    added = site_record["site_cls"]["added"]
    assert len(core) == N_CORE_SITES
    assert len(added) == N_SITES - N_CORE_SITES
    assert set(core).isdisjoint(added)


def test_draw_allocations_patches_per_class(frame: pd.DataFrame):
    """Every allocation gives every class exactly 160 patches (Table allocations)."""
    frames, _ = draw_allocations(frame, ["site_cls", "bg_cls"], ["site_cls"], 0, 0)
    for allocation in ALLOCATIONS:
        counts = frames[allocation].groupby("cancer_type").size()
        assert counts["site_cls"] == 160
        assert counts["bg_cls"] == 160


def test_draw_allocations_nesting(frame: pd.DataFrame):
    """Patient nesting across allocations follows Eq. (nesting)."""
    frames, _ = draw_allocations(frame, ["site_cls", "bg_cls"], ["site_cls"], 0, 0)

    def patients(name: str) -> set[str]:
        sub = frames[name]
        return set(sub.loc[sub["cancer_type"] == "site_cls", "case_id"])

    deep, broad5, broad10 = patients("deep"), patients("broad5"), patients("broad10")
    assert deep <= broad5
    assert deep <= broad10
    assert len(broad5 & broad10) == 10
    assert len(deep) == 5
    assert len(broad5) == 20
    assert len(broad10) == 20


def test_draw_allocations_deep_prefix_nesting(frame: pd.DataFrame):
    """A shared patient's broad5 patches are the first 8 of its deep 32 patches."""
    frames, _ = draw_allocations(frame, ["site_cls", "bg_cls"], ["site_cls"], 0, 0)
    deep = frames["deep"]
    broad5 = frames["broad5"]
    shared_patient = deep.loc[deep["cancer_type"] == "site_cls", "case_id"].iloc[0]

    deep_patches = deep.loc[deep["case_id"] == shared_patient, "patch_id"].tolist()
    broad5_patches = broad5.loc[broad5["case_id"] == shared_patient, "patch_id"].tolist()
    assert deep_patches[:8] == broad5_patches


def test_draw_allocations_background_matches_grid_recipe(frame: pd.DataFrame):
    """Background classes reuse the exp-5 G10_m16 draw recipe unchanged."""
    frames, _ = draw_allocations(frame, ["site_cls", "bg_cls"], ["site_cls"], 0, 0)
    expected = sample_cell_draw(
        frame, ["site_cls", "bg_cls"], *BACKGROUND_CELL, split_index=0, draw_index=0
    )
    expected_bg = expected.loc[expected["cancer_type"] == "bg_cls", "patch_id"].tolist()
    for allocation in ALLOCATIONS:
        actual_bg = frames[allocation]
        actual_bg = actual_bg.loc[actual_bg["cancer_type"] == "bg_cls", "patch_id"].tolist()
        assert sorted(actual_bg) == sorted(expected_bg)


def test_draw_allocations_deterministic(frame: pd.DataFrame):
    """Repeated calls with the same arguments draw identical allocations."""
    frames_a, record_a = draw_allocations(frame, ["site_cls", "bg_cls"], ["site_cls"], 1, 2)
    frames_b, record_b = draw_allocations(frame, ["site_cls", "bg_cls"], ["site_cls"], 1, 2)
    for allocation in ALLOCATIONS:
        assert frames_a[allocation]["patch_id"].tolist() == frames_b[allocation]["patch_id"].tolist()
    assert record_a == record_b
