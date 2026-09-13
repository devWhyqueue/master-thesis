"""Unit tests for patient-coverage seed, neighbour, random, and coverage logic."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from breadth.sampling import sample_class_patches

from neighbours import ALLOCATIONS, DEEP_CELL
from neighbours.allocation import (
    allocation_frames,
    neighbour_patients,
    random_patients,
    seed_patients,
)
from neighbours.coverage import _coverage_distance, kappa as _kappa
from neighbours.embedding import cosine_distances, embed

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


def test_seed_patients_matches_exp5_draw(frame: pd.DataFrame):
    """Seeds are the ordered unique patients of the exp-5 five-patient draw."""
    seeds = seed_patients(frame, ["cls"], 0, split=0, draw=0)
    expected_indices = sample_class_patches(frame, ["cls"], 0, DEEP_CELL, (0, 0))
    expected_cases = frame.loc[expected_indices, "case_id"].astype(str).tolist()
    expected_order = list(dict.fromkeys(expected_cases))
    assert seeds == expected_order
    assert len(seeds) == 5


def test_neighbour_patients_tie_break_picks_lexicographic_id():
    """Each seed's tie among equidistant candidates breaks toward the smaller id."""
    seeds = ["s0", "s1"]
    pool = ["s0", "s1", "a", "b", "c", "d", "e", "f"]
    dist = np.array(
        [
            #  s0    s1    a     b     c     d     e     f
            [0.0, 9.0, 0.1, 0.1, 0.5, 0.6, 0.7, 0.8],
            [9.0, 0.0, 0.6, 0.7, 0.05, 0.2, 0.3, 0.4],
        ]
    )
    selected = neighbour_patients(seeds, pool, dist, np.random.default_rng(0))
    first_round_picks = {seed: patient for patient, seed in selected[:2]}
    # s0 ties between "a" and "b"; the tie-break favours the smaller id.
    assert first_round_picks["s0"] == "a"
    assert first_round_picks["s1"] == "c"


def test_neighbour_patients_deterministic():
    seeds = ["s0", "s1"]
    pool = ["s0", "s1", "a", "b", "c", "d", "e", "f"]
    dist = np.array(
        [[0.0, 9.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], [9.0, 0.0, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]]
    )
    a = neighbour_patients(seeds, pool, dist, np.random.default_rng(7))
    b = neighbour_patients(seeds, pool, dist, np.random.default_rng(7))
    assert a == b
    assert len(a) == 6  # N_ROUNDS (3) * len(seeds) (2)
    assert set(seeds).isdisjoint({patient for patient, _selector in a})


def test_random_patients_excludes_seeds_and_is_deterministic():
    seeds = ["s0", "s1"]
    pool = ["s0", "s1"] + [f"p{i:02d}" for i in range(20)]
    chosen_a = random_patients(seeds, pool, np.random.default_rng(3))
    chosen_b = random_patients(seeds, pool, np.random.default_rng(3))
    assert chosen_a == chosen_b
    assert len(chosen_a) == 15
    assert set(chosen_a).isdisjoint(seeds)


def test_allocation_frames_nesting_and_patch_counts(frame: pd.DataFrame):
    """Patient nesting (Eq. nesting) and 160 patches per class in every allocation."""
    class_names = ["cls"]
    seeds = seed_patients(frame, class_names, 0, split=0, draw=0)
    pool = sorted(frame["case_id"].astype(str).unique())
    dist = np.zeros((len(seeds), len(pool)))
    neighbours = neighbour_patients(seeds, pool, dist, np.random.default_rng(1))
    randoms = random_patients(seeds, pool, np.random.default_rng(2))
    record = {"cls": {"seeds": seeds, "neighbours": neighbours, "random": randoms}}

    frames = allocation_frames(frame, class_names, record)
    deep_patients = set(seeds)
    n_patients = deep_patients | {patient for patient, _selector in neighbours}
    r_patients = deep_patients | set(randoms)
    assert deep_patients <= n_patients
    assert deep_patients <= r_patients
    assert len(n_patients) == 20
    assert len(r_patients) == 20
    for allocation in ALLOCATIONS:
        assert len(frames[allocation]) == 160


def test_broad_seed_patches_are_deep_prefix(frame: pd.DataFrame):
    """A seed's broad 8 patches are the first 8 of its 32 deep patches."""
    class_names = ["cls"]
    seeds = seed_patients(frame, class_names, 0, split=0, draw=0)
    pool = sorted(frame["case_id"].astype(str).unique())
    dist = np.zeros((len(seeds), len(pool)))
    neighbours = neighbour_patients(seeds, pool, dist, np.random.default_rng(1))
    randoms = random_patients(seeds, pool, np.random.default_rng(2))
    record = {"cls": {"seeds": seeds, "neighbours": neighbours, "random": randoms}}
    frames = allocation_frames(frame, class_names, record)

    deep_indices = sample_class_patches(frame, class_names, 0, DEEP_CELL, (0, 0))
    deep = frame.loc[deep_indices]
    seed = seeds[0]
    deep_patches = deep.loc[deep["case_id"].astype(str) == seed, "patch_id"].tolist()

    broad = frames["neighbours"]
    broad_patches = broad.loc[broad["case_id"].astype(str) == seed, "patch_id"].tolist()
    assert deep_patches[:8] == broad_patches


def test_allocation_frames_deterministic(frame: pd.DataFrame):
    """Repeated calls with the same record draw identical patch allocations."""
    class_names = ["cls"]
    seeds = seed_patients(frame, class_names, 0, split=0, draw=0)
    pool = sorted(frame["case_id"].astype(str).unique())
    dist = np.zeros((len(seeds), len(pool)))
    neighbours = neighbour_patients(seeds, pool, dist, np.random.default_rng(4))
    randoms = random_patients(seeds, pool, np.random.default_rng(5))
    record = {"cls": {"seeds": seeds, "neighbours": neighbours, "random": randoms}}
    frames_a = allocation_frames(frame, class_names, record)
    frames_b = allocation_frames(frame, class_names, record)
    for allocation in ALLOCATIONS:
        assert (
            frames_a[allocation]["patch_id"].tolist() == frames_b[allocation]["patch_id"].tolist()
        )


def test_embed_unit_norm_and_self_distance_zero():
    """embed centres and L2-normalises; a patient's cosine distance to itself is 0."""
    means = {
        ("p0", "c"): np.array([3.0, 0.0, 0.0]),
        ("p1", "c"): np.array([0.0, 4.0, 0.0]),
    }
    mu = np.array([1.0, 1.0, 0.0])
    embedded = embed(means, mu)
    for vec in embedded.values():
        assert np.linalg.norm(vec) == pytest.approx(1.0)
    e = np.stack([embedded[("p0", "c")]])
    assert cosine_distances(e, e)[0, 0] == pytest.approx(0.0, abs=1e-10)


def test_coverage_distance_and_kappa_toy_example():
    """Coverage distance (Eq. coverage) and leakage kappa (Eq. leakage) on a toy case."""
    # One validation patient sits closer to the "random" candidate than to the
    # "neighbour" candidate, which in turn is closer than the lone deep seed.
    val = np.array([[1.0, 0.0]])
    seed = np.array([[0.0, 1.0]])
    neighbour_set = np.array([[0.0, 1.0], [0.7, 0.3]])
    random_set = np.array([[0.0, 1.0], [0.99, 0.02]])

    def _dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a = a / np.linalg.norm(a, axis=1, keepdims=True)
        b = b / np.linalg.norm(b, axis=1, keepdims=True)
        return cosine_distances(a, b)

    r_deep = _coverage_distance(val, seed)
    r_neighbours = float(np.mean(np.min(_dist(val, neighbour_set), axis=1)))
    r_random = float(np.mean(np.min(_dist(val, random_set), axis=1)))
    # Recompute r_deep with normalisation for a fair, consistent comparison.
    r_deep = float(np.mean(np.min(_dist(val, seed), axis=1)))

    assert r_deep > r_neighbours > r_random
    kappa = _kappa(r_deep, r_neighbours, r_random)
    assert 0.0 < kappa < 1.0
    assert _kappa(1.0, 1.0, 0.0) == pytest.approx(0.0)
    assert np.isnan(_kappa(1.0, 0.5, 1.0))
