"""Unit tests for shortage-selection allocation, classification, and sharding logic."""

from __future__ import annotations

import pandas as pd
import pytest

from neighbours.stages import decode_shard_index

from shortage import ALLOCATIONS
from shortage.allocation import allocation_frames
from shortage.analyze import classify
from shortage.census import _ratios

N_PATCHES = 32
N_POOL_PATIENTS = 10


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


@pytest.fixture
def frame() -> pd.DataFrame:
    rows: list[dict] = []
    for patient in range(N_POOL_PATIENTS):
        rows.extend(_patient_rows("cls", f"TCGA-01-{patient:04d}"))
    return pd.DataFrame(rows)


def test_allocation_frames_160_patches_and_focal_present(frame: pd.DataFrame):
    """Every allocation contributes 160 patches per class and includes the focal patient."""
    class_names = ["cls"]
    patients = sorted(frame["case_id"].astype(str).unique())
    focal = patients[0]
    record = {
        "cls": {
            "seeds": [focal],
            "focal": focal,
            "random": [focal] + patients[1:5],
            "dispersed": [focal] + patients[5:9],
        }
    }

    frames = allocation_frames(frame, class_names, record)
    for allocation in ALLOCATIONS:
        assert len(frames[allocation]) == 160
        assert focal in frames[allocation]["case_id"].astype(str).unique()


@pytest.mark.parametrize(
    ("sel_ci", "expected"),
    [
        ((1.5, 2.0), "selection_mitigates_shortage"),
        ((-0.5, 0.5), "no_practical_gain"),
        ((-2.0, -1.5), "selection_harms"),
        ((0.5, 1.5), "inconclusive"),  # straddles the threshold from below
        ((-2.0, 0.5), "inconclusive"),  # straddles the threshold from above zero
    ],
)
def test_classify_truth_table(sel_ci, expected):
    assert classify(sel_ci) == expected


def test_rho_sel_ratio():
    """rho_sel = (r_5 - r_dispersed) / (r_5 - r_ran20) (Eq. ratio)."""
    r = {"r_5": 0.386, "r_dispersed": 0.324, "r_ran20": 0.294}
    ratios = _ratios(r)
    expected = (0.386 - 0.324) / (0.386 - 0.294)
    assert ratios["rho_sel"] == pytest.approx(expected)


def test_decode_shard_index_accepts_5_rejects_6():
    """The exp-8 shard decoder generalises to exp-12's 2-allocation, 3-split grid."""
    names = tuple(ALLOCATIONS)
    s_idx, allocation = decode_shard_index(5, names)
    assert s_idx == 2
    assert allocation == names[1]
    with pytest.raises(ValueError):
        decode_shard_index(6, names)
