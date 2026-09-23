"""Regression test: every gate's own return dict must be JSON-serializable as-is. numpy scalar
types (np.bool_, np.float64 wrapped in a bool comparison) silently reach json.dump and crash it;
plain unit tests that monkeypatch whole gate functions never exercise the real arithmetic that
produces them, so this drives the actual gate4/gate5 logic on synthetic per-(split, draw) data.
"""

from __future__ import annotations

import json

import joint.gating.gate as gate
import joint.gating.robustness as robustness


def test_gate4_joint_rescue_result_is_json_serializable(monkeypatch):
    # Alternating positive/negative per-split damage fall, so splits_positive's sum-and-compare
    # path (the one that leaked a numpy bool) actually executes on real numpy reductions.
    values = {0: 3.0, 1: -1.0, 2: 2.0}

    def fake_rescue_contrast(config, split_idx, draw_idx):
        return {
            "damage_fall_pp": values[split_idx],
            "r_rise_pp": 2.5,
            "b_drop_pp": 0.2,
        }

    monkeypatch.setattr(gate, "rescue_contrast", fake_rescue_contrast)
    result = gate.gate4_joint_rescue({"dataset": {"name": "bracs"}})
    json.dumps(result)  # raises TypeError if any value is a non-native numpy type


def test_gate5_robustness_result_is_json_serializable(monkeypatch):
    monkeypatch.setattr(robustness, "_boundary_check", lambda config: ([], []))
    monkeypatch.setattr(robustness, "_fixed_lambda_rescue", lambda config: 1.5)
    result = robustness.gate5_robustness({"dataset": {"name": "bracs"}})
    json.dumps(result)
