from __future__ import annotations

import pytest

from derive_deficit_thresholds import STABILITY_FLOOR, DispersionRow, report_thresholds


def test_thresholds_are_grouped_per_dataset_not_pooled_across_the_noisiest_one():
    """A quiet dataset's threshold must come from its own sigma, not BRACS's.

    Regression for the published bug: a pooled global max calibrated
    TCGA-UT's gate to BRACS's noise floor.
    """
    rows = [
        DispersionRow("bracs", 0, None, "ba", sigma_seed=0.02),
        DispersionRow("tcga_ut", 0, None, "ba", sigma_seed=0.001),
    ]

    thresholds = report_thresholds(rows)

    assert thresholds["bracs"].discrimination_threshold == pytest.approx(0.04)
    assert thresholds["tcga_ut"].discrimination_threshold == pytest.approx(
        STABILITY_FLOOR
    )
