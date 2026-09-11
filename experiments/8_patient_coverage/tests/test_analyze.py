"""Unit tests for the patient-coverage interpretation classifier and tertiles."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neighbours.analyze import classify
from neighbours.secondary import _tertile_rows


@pytest.mark.parametrize(
    ("dc_ci", "bn_ci", "expected"),
    [
        ((1.5, 2.0), (-0.5, 0.5), "patient_coverage"),
        ((1.5, 2.0), (-3.0, -1.5), "patient_coverage"),  # a negative b_N still counts
        ((-0.5, 0.5), (1.5, 2.0), "count_beyond_coverage"),
        ((1.5, 2.0), (1.2, 1.8), "both"),
        ((0.5, 1.5), (-0.5, 0.5), "inconclusive"),  # coverage gain straddles threshold
        ((-0.5, 0.5), (0.5, 1.5), "inconclusive"),  # residual straddles threshold
        ((-2.0, 0.5), (-0.5, 0.5), "inconclusive"),  # coverage gain extends below -1
    ],
)
def test_classify_truth_table(dc_ci, bn_ci, expected):
    assert classify(dc_ci, bn_ci) == expected


def test_tertile_rows_split_sizes():
    """Ten test patients split into near/middle/far thirds via a stable argsort."""
    n = 10
    case_ids = [f"p{i}" for i in range(n)]
    distances = {case: float(i) for i, case in enumerate(case_ids)}
    # recall equals the patient's distance rank (as a fraction), so each
    # tertile's mean directly reveals which -- and how many -- patients it holds.
    pairs_n = pd.DataFrame(
        {"case_id": case_ids, "recall": [i / 100.0 for i in range(n)], "weight": [1.0] * n}
    )
    pairs_r = pd.DataFrame(
        {"case_id": case_ids, "recall": [0.6] * n, "weight": [1.0] * n}
    )

    rows = _tertile_rows(pairs_n, pairs_r, distances)
    assert len(rows) == 3

    expected_groups = np.array_split(np.arange(n), 3)
    assert [len(g) for g in expected_groups] == [4, 3, 3]
    for row, group in zip(rows, expected_groups):
        assert row["recall_neighbours"] == pytest.approx(float(np.mean(group)))
        assert row["recall_random"] == pytest.approx(60.0)
