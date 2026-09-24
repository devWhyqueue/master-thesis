"""Unit tests for exp-39's phase-01 schedule: patient cohort, permutation, and patch identity."""

from __future__ import annotations

import pandas as pd
import pytest

from prevalence import BALANCED, DEPTH
from prevalence.fit import class_permutation

from transfer import ARMS
from transfer.schedule import draw_schedule

_NAMES = ["clsA", "clsB", "clsC"]
_G = 3


def _synthetic_train_df() -> pd.DataFrame:
    """More than G eligible patients per class, each with exactly DEPTH patches."""
    rows = []
    for name in _NAMES:
        for p in range(6):
            case = f"{name}-P{p}"
            for patch in range(DEPTH):
                rows.append(
                    {
                        "cancer_type": name,
                        "case_id": case,
                        "slide_id": f"{case}-S0",
                        "patch_id": f"{case}-p{patch:04d}",
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def train_df() -> pd.DataFrame:
    return _synthetic_train_df()


def _cell(train_df: pd.DataFrame, draw_idx: int = 10, split_idx: int = 0) -> dict:
    return draw_schedule(train_df, _NAMES, split_idx=split_idx, draw_idx=draw_idx, g=_G)


def test_every_arm_present(train_df) -> None:
    assert set(_cell(train_df)["arms"]) == set(ARMS)


def test_class_permutation_matches_prevalence(train_df) -> None:
    cell = _cell(train_df)
    perm = class_permutation(split_idx=0, draw_idx=10, num_classes=len(_NAMES))
    assert cell["class_permutation"] == [_NAMES[i] for i in perm]


def test_patients_fixed_across_arms(train_df) -> None:
    cell = _cell(train_df)
    for name in _NAMES:
        assert len(cell["patients"][name]) == _G


def test_b_arm_is_balanced(train_df) -> None:
    b = _cell(train_df)["arms"]["B"]
    assert b["prior_arm"] is None
    assert all(c == _G * BALANCED for c in b["class_counts"].values())
    assert b["n_rows"] == _G * BALANCED * len(_NAMES)
    assert len(b["patch_identity"]) == b["n_rows"]
    assert len({tuple(row) for row in b["patch_identity"]}) == b["n_rows"]


def test_p_arm_reuses_b_rows_reweighted_toward_r_shares(train_df) -> None:
    arms = _cell(train_df)["arms"]
    b, p10, r10 = arms["B"], arms["P10"], arms["R10"]
    assert p10["data_arm"] == "r1"
    assert p10["prior_arm"] == "r10"
    assert p10["patch_identity"] == b["patch_identity"]
    assert p10["prior_counts"] == r10["class_counts"]


def test_s_arm_reuses_r_rows_at_uniform_prior(train_df) -> None:
    arms = _cell(train_df)["arms"]
    r10, s10 = arms["R10"], arms["S10"]
    assert s10["data_arm"] == "r10"
    assert s10["prior_arm"] == "r1"
    assert s10["patch_identity"] == r10["patch_identity"]


def test_schedule_deterministic_across_calls(train_df) -> None:
    a = _cell(train_df, draw_idx=15, split_idx=1)
    b = _cell(train_df, draw_idx=15, split_idx=1)
    assert a == b


def test_distinct_draws_give_distinct_patients(train_df) -> None:
    a = _cell(train_df, draw_idx=10)
    b = _cell(train_df, draw_idx=11)
    assert a["patients"] != b["patients"]
