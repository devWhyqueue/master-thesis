"""Regression test: gate 5's fixed-lambda rescue must only read allocations that fit actually
writes. native_B has no fixed-lambda variant (its own tuned lambda already is the fixed one,
PLAN.md line 39), so reading "native_B_fixedlambda" would always raise on a real pilot.
"""

from __future__ import annotations

from joint import FIXED_LAMBDA_ARMS_BY_SETTING
from joint.gating import robustness


def test_fixed_lambda_rescue_only_reads_allocations_fit_actually_writes(monkeypatch):
    written = {
        f"{setting}_{arm}_fixedlambda" for setting, arms in FIXED_LAMBDA_ARMS_BY_SETTING.items() for arm in arms
    } | {"native_B", "joint_B"}  # native_B/joint_B: the tuned fits every fixed-lambda pass reuses

    queried: set[str] = set()

    def fake_test_ba(config, allocation, split_idx, draw_idx):
        queried.add(allocation)
        return 50.0

    monkeypatch.setattr(robustness, "test_ba", fake_test_ba)
    robustness._fixed_lambda_rescue({"dataset": {"name": "bracs"}})

    assert queried <= written, f"read allocation(s) fit never writes: {queried - written}"
