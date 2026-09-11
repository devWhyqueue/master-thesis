"""Unit tests for the site-coverage interpretation classifier."""

from __future__ import annotations

import pytest

from sites.analyze import classify


@pytest.mark.parametrize(
    ("ds_ci", "bw_ci", "expected"),
    [
        ((1.5, 2.0), (-0.5, 0.5), "site_coverage"),
        ((-0.5, 0.5), (1.5, 2.0), "within_site_patient_variation"),
        ((1.5, 2.0), (1.2, 1.8), "both"),
        ((0.5, 1.5), (-0.5, 0.5), "inconclusive"),  # site gain straddles the threshold
        ((-0.5, 0.5), (0.5, 1.5), "inconclusive"),  # residual straddles the threshold
        ((-2.0, 0.5), (-0.5, 0.5), "inconclusive"),  # site gain interval extends below -1
    ],
)
def test_classify_truth_table(ds_ci, bw_ci, expected):
    assert classify(ds_ci, bw_ci) == expected
