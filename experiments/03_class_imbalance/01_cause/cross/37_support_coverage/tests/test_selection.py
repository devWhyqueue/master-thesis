"""Toy two-cluster pool (PLAN.md "Unit tests"): the coverage pick spans both clusters, the
redundant pick takes the centre, and every pick respects its per-patient quota.
"""

from __future__ import annotations

import numpy as np
import pytest

from support.coverage import patient_balanced_mean
from support.selection import greedy_quota_coverage, prefix_indices, redundant_pick

# patient 0: row 0 (cluster A, angle 0 deg) and row 1 (near the pool centre, angle ~45 deg).
# patient 1: row 2 (near the pool centre, angle ~51 deg) and row 3 (cluster B, angle 90 deg).
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


def test_greedy_quota_coverage_spans_both_clusters():
    selected = greedy_quota_coverage(_X, _PATIENT_IDX, _QUOTA)
    assert set(selected.tolist()) == {0, 3}


def test_redundant_pick_takes_the_centre():
    centre = patient_balanced_mean(_X, _PATIENT_IDX)
    selected = redundant_pick(_X, _PATIENT_IDX, _QUOTA, centre)
    assert set(selected.tolist()) == {1, 2}


def test_prefix_indices_takes_each_patients_first_rows():
    selected = prefix_indices(_PATIENT_IDX, _QUOTA)
    assert set(selected.tolist()) == {0, 2}


@pytest.mark.parametrize("arm", ["coverage", "redundant", "prefix"])
def test_quotas_are_respected(arm: str):
    if arm == "coverage":
        selected = greedy_quota_coverage(_X, _PATIENT_IDX, _QUOTA)
    elif arm == "redundant":
        selected = redundant_pick(
            _X, _PATIENT_IDX, _QUOTA, patient_balanced_mean(_X, _PATIENT_IDX)
        )
    else:
        selected = prefix_indices(_PATIENT_IDX, _QUOTA)
    counts = np.bincount(_PATIENT_IDX[selected], minlength=2)
    assert counts.tolist() == _QUOTA


def test_greedy_quota_coverage_raises_when_quota_exceeds_pool():
    with pytest.raises(RuntimeError):
        greedy_quota_coverage(_X, _PATIENT_IDX, [3, 1])
