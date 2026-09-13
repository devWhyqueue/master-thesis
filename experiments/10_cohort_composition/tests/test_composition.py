"""Unit tests for cohort-composition allocation, classification, and sharding logic."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neighbours.stages import decode_shard_index

from composition import ALLOCATIONS
from composition.allocation import allocation_frames, clustered_patients, dispersed_patients
from composition.analyze import classify

N_PATCHES = 32
N_POOL_PATIENTS = 25


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


def test_clustered_patients_excludes_only_focal_tie_break_by_id():
    """Clustered keeps every non-focal candidate in play; ties break toward the smaller id."""
    focal = "f0"
    others = [chr(ord("a") + i) for i in range(20)]  # a..t, 20 candidates
    pool = [focal] + others
    n = len(pool)
    dist_pool = np.zeros((n, n))
    distances = [0.1 * i for i in range(18)] + [1.8, 1.8]  # "s" and "t" tie for 19th
    for j, d in enumerate(distances):
        dist_pool[0, j + 1] = d

    chosen = clustered_patients(focal, pool, dist_pool)
    assert len(chosen) == 20
    assert chosen[0] == focal
    assert "s" in chosen
    assert "t" not in chosen


def test_dispersed_patients_two_clusters_first_pick_from_other_cluster():
    """Facility location's first addition covers the cluster the focal patient misses."""
    cluster_a = [f"a{i}" for i in range(10)]
    cluster_b = [f"b{i}" for i in range(10)]
    pool = cluster_a + cluster_b
    focal = cluster_a[0]
    n = len(pool)
    dist_pool = np.full((n, n), 1.0)
    dist_pool[:10, :10] = 0.01
    dist_pool[10:, 10:] = 0.01
    np.fill_diagonal(dist_pool, 0.0)

    chosen = dispersed_patients(focal, pool, dist_pool)
    assert len(chosen) == 20
    assert set(chosen) == set(pool)
    assert chosen[0] == focal
    assert chosen[1] in cluster_b


def test_allocation_frames_160_patches_and_focal_present(frame: pd.DataFrame):
    """Every allocation contributes 160 patches per class and includes the focal patient."""
    class_names = ["cls"]
    patients = sorted(frame["case_id"].astype(str).unique())
    focal = patients[0]
    record = {
        "cls": {
            "seeds": [focal],
            "focal": focal,
            "clustered": [focal] + patients[1:20],
            "random": [focal] + patients[2:21],
            "dispersed": [focal] + patients[5:24],
        }
    }

    frames = allocation_frames(frame, class_names, record)
    for allocation in ALLOCATIONS:
        assert len(frames[allocation]) == 160
        assert focal in frames[allocation]["case_id"].astype(str).unique()


@pytest.mark.parametrize(
    ("con_ci", "sel_ci", "expected"),
    [
        ((1.5, 2.0), (1.5, 2.0), "selection_beyond_sampling"),
        ((1.5, 2.0), (-0.5, 0.5), "concentration_damage"),
        ((1.5, 2.0), (0.5, 1.5), "concentration_damage"),  # sel straddles, not "above"
        ((-0.5, 0.5), (-0.5, 0.5), "no_composition_effect"),
        ((-0.5, 0.5), (1.5, 2.0), "inconclusive"),  # con within, sel above
        ((0.5, 1.5), (-0.5, 0.5), "inconclusive"),  # con straddles the threshold
        ((-2.0, 0.5), (-0.5, 0.5), "inconclusive"),  # con extends below -1
    ],
)
def test_classify_truth_table(con_ci, sel_ci, expected):
    assert classify(con_ci, sel_ci) == expected


def test_decode_shard_index_accepts_shard_8_for_three_allocations():
    """The exp-8 shard decoder generalises past its own 2-allocation constant."""
    names = tuple(ALLOCATIONS)
    s_idx, allocation = decode_shard_index(8, names)
    assert s_idx == 2
    assert allocation == names[2]
    with pytest.raises(ValueError):
        decode_shard_index(9, names)
