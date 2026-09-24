"""Coverage deficit and centre error (PLAN.md "Measures") on the same toy two-cluster pool."""

from __future__ import annotations

import numpy as np

from support.coverage import (
    centre_error,
    coverage_deficit,
    median_pairwise_distance,
    normalize,
    patient_balanced_mean,
)
from support.selection import greedy_quota_coverage, redundant_pick

_X = np.array(
    [
        [1.0, 0.0],
        [1.0, 1.0],
        [0.9, 1.1],
        [0.0, 1.0],
    ]
)
_PATIENT_IDX = np.array([0, 0, 1, 1])
_QUOTA = [1, 1]


def test_normalize_unit_rows():
    e = normalize(_X)
    np.testing.assert_allclose(np.linalg.norm(e, axis=1), 1.0)


def test_coverage_pick_covers_better_than_redundant_pick():
    centre = patient_balanced_mean(_X, _PATIENT_IDX)
    coverage_pick = greedy_quota_coverage(_X, _PATIENT_IDX, _QUOTA)
    redundant = redundant_pick(_X, _PATIENT_IDX, _QUOTA, centre)
    assert coverage_deficit(_X, coverage_pick) <= coverage_deficit(_X, redundant)


def test_redundant_pick_is_closer_to_centre():
    centre = patient_balanced_mean(_X, _PATIENT_IDX)
    coverage_pick = greedy_quota_coverage(_X, _PATIENT_IDX, _QUOTA)
    redundant = redundant_pick(_X, _PATIENT_IDX, _QUOTA, centre)
    assert centre_error(_X, redundant, centre) <= centre_error(_X, coverage_pick, centre)


def test_median_pairwise_distance_positive():
    assert median_pairwise_distance(_X) > 0.0
