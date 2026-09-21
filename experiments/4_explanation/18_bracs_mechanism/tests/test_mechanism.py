"""Unit tests for the BRACS mechanism arm set and derived shares/ratios/interaction."""

from __future__ import annotations

import numpy as np
import pytest

from centre import PATIENT_COUNTS

from mechanism import ARMS, CENTRE_FAMILIES
from mechanism.analyze import derived


def test_arms_has_eighteen_unique_entries():
    """ARMS covers R, C, N, CW, RW, RWc at 5, 10, and 20 patients, with no duplicates."""
    assert len(ARMS) == 18
    assert len(set(ARMS)) == 18
    assert CENTRE_FAMILIES == ("R", "C", "N", "CW")


def test_derived_on_additive_toy_gaps_gives_zero_interaction():
    """Constant per-family offsets from R give clean shares, unit noise ratio, and zero interaction."""
    base = {
        "R": (60.0, 66.0, 70.0),
        "C": (70.0, 73.0, 75.0),
        "N": (61.0, 67.0, 71.0),
        "CW": (72.0, 75.0, 77.0),
        "RW": (62.0, 68.0, 72.0),
        "RWc": (61.0, 67.0, 71.0),
    }
    arm = {
        f"{f}{g}": np.array([v])
        for f, vals in base.items()
        for g, v in zip(PATIENT_COUNTS, vals)
    }

    out = derived(arm)

    assert out["share_C_5_to_10"][0] == pytest.approx(0.5)
    assert out["share_CW_5_to_10"][0] == pytest.approx(0.5)
    assert out["share_W_5_to_10"][0] == pytest.approx(0.0)
    assert out["share_interaction_5_to_10"][0] == pytest.approx(0.0)
    for g in PATIENT_COUNTS:
        assert out[f"interaction_{g}"][0] == pytest.approx(0.0)
    for step in ("5_to_10", "5_to_20", "10_to_20"):
        assert out[f"noise_ratio_{step}"][0] == pytest.approx(1.0)
    assert out["C5_minus_R20"][0] == pytest.approx(0.0)
