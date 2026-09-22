"""Unit tests for concentrated-coverage focal-seed, allocation, and strata logic."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from breadth.sampling import sample_class_patches

from neighbours import DEEP_CELL
from neighbours.allocation import draw_rng, neighbour_patients, random_patients, seed_patients
from neighbours.secondary import _group_rows

from concentrated import ALLOCATIONS
from concentrated.allocation import (
    allocation_frames,
    concentrated_patients,
    exp8_random_patients,
    focal_seed,
)

N_PATCHES = 32
N_POOL_PATIENTS = 25  # >= 5 seeds + 15 added, comfortably above both draw sizes


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


def test_focal_seed_deterministic_and_one_of_the_seeds():
    seeds = ["s0", "s1", "s2", "s3", "s4"]
    a = focal_seed(seeds, split=0, draw=0, c_idx=0)
    b = focal_seed(seeds, split=0, draw=0, c_idx=0)
    assert a == b
    assert 0 <= a < len(seeds)
    assert seeds[a] in seeds


def test_concentrated_patients_excludes_seeds_nearest_tie_break_by_id():
    seeds = ["s0", "s1"]
    letters = [chr(ord("a") + i) for i in range(15)]
    pool = seeds + letters
    # "a" and "b" tie for nearest to the focal seed (s0); the rest are farther
    # and strictly ordered, so all 15 letters are chosen, "a" then "b" first.
    dist = np.array(
        [[0.0, 9.0] + [0.1, 0.1] + [0.2 + 0.1 * i for i in range(13)], [9.0] * 17]
    )
    chosen = concentrated_patients(0, seeds, pool, dist)
    assert len(chosen) == 15
    assert set(chosen).isdisjoint(seeds)
    assert chosen[0] == "a"
    assert chosen[1] == "b"


def test_exp8_random_patients_replays_exp8_sequence():
    """Replays exp-8's neighbour-selection burn, then its random draw, bit-for-bit."""
    seeds = ["s0", "s1"]
    pool = ["s0", "s1"] + [f"p{i:02d}" for i in range(20)]
    dist = np.zeros((len(seeds), len(pool)))

    rng_expected = draw_rng(0, 0, 0)
    neighbour_patients(seeds, pool, dist, rng_expected)
    expected = random_patients(seeds, pool, rng_expected)

    actual = exp8_random_patients(seeds, pool, split=0, draw=0, c_idx=0)
    assert actual == expected


def test_allocation_frames_nesting_and_patch_counts(frame: pd.DataFrame):
    """Patient nesting (Eq. nesting) and 160 patches per class in every allocation."""
    class_names = ["cls"]
    seeds = seed_patients(frame, class_names, 0, split=0, draw=0)
    pool = sorted(frame["case_id"].astype(str).unique())
    dist = np.zeros((len(seeds), len(pool)))
    concentrated = concentrated_patients(0, seeds, pool, dist)
    randoms = exp8_random_patients(seeds, pool, split=0, draw=0, c_idx=0)
    record = {
        "cls": {
            "seeds": seeds,
            "focal": seeds[0],
            "concentrated": concentrated,
            "random": randoms,
            "focal_cell": [],
        }
    }

    frames = allocation_frames(frame, class_names, record)
    deep_patients = set(seeds)
    c_patients = deep_patients | set(concentrated)
    r_patients = deep_patients | set(randoms)
    assert deep_patients <= c_patients
    assert deep_patients <= r_patients
    assert len(c_patients) == 20
    assert len(r_patients) == 20
    for allocation in ALLOCATIONS:
        assert len(frames[allocation]) == 160


def test_broad_seed_patches_are_deep_prefix(frame: pd.DataFrame):
    """A seed's broad 8 patches are the first 8 of its 32 deep patches."""
    class_names = ["cls"]
    seeds = seed_patients(frame, class_names, 0, split=0, draw=0)
    pool = sorted(frame["case_id"].astype(str).unique())
    dist = np.zeros((len(seeds), len(pool)))
    concentrated = concentrated_patients(0, seeds, pool, dist)
    randoms = exp8_random_patients(seeds, pool, split=0, draw=0, c_idx=0)
    record = {
        "cls": {
            "seeds": seeds,
            "focal": seeds[0],
            "concentrated": concentrated,
            "random": randoms,
            "focal_cell": [],
        }
    }
    frames = allocation_frames(frame, class_names, record)

    deep_indices = sample_class_patches(frame, class_names, 0, DEEP_CELL, (0, 0))
    deep = frame.loc[deep_indices]
    seed = seeds[0]
    deep_patches = deep.loc[deep["case_id"].astype(str) == seed, "patch_id"].tolist()

    broad = frames["concentrated"]
    broad_patches = broad.loc[broad["case_id"].astype(str) == seed, "patch_id"].tolist()
    assert deep_patches[:8] == broad_patches


def test_group_rows_nan_for_empty_group():
    """An empty focal-cell group gets NaN rather than a ZeroDivisionError."""
    pairs_low = pd.DataFrame(
        {"case_id": ["p0", "p1"], "recall": [0.5, 0.7], "weight": [0.5, 0.5]}
    )
    pairs_r = pd.DataFrame(
        {"case_id": ["p0", "p1"], "recall": [0.4, 0.6], "weight": [0.5, 0.5]}
    )
    groups = [np.array([], dtype=int), np.array([0, 1])]
    rows = _group_rows(pairs_low, pairs_r, groups, low="concentrated")
    assert len(rows) == 2
    assert np.isnan(rows[0]["recall_concentrated"])
    assert np.isnan(rows[0]["recall_random"])
    assert np.isnan(rows[0]["gain"])
    assert not np.isnan(rows[1]["recall_concentrated"])
