"""Unit tests for the site-coverage interpretation classifier and payload."""

from __future__ import annotations

import numpy as np
import pytest

from sites.analyze import classify
from sites.results import allocation_payload


def test_allocation_payload_draw_contrasts():
    points = {
        "deep": np.array([50.0, 52.0]),
        "broad5": np.array([60.0, 61.0]),
        "broad10": np.array([61.0, 60.5]),
    }
    dists = {name: np.full(3, p.mean()) for name, p in points.items()}
    payload = allocation_payload(dists, dists, points)
    gain = payload["draw_contrasts"]["site_gain"]
    assert gain["values"] == [1.0, -0.5]
    assert gain["n_positive"] == 1
    assert gain["sd"] == pytest.approx(0.75)
    assert payload["site_class"]["deep"]["draw_dispersion"] == pytest.approx(1.0)


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
