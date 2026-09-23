"""Unit test: a failed pilot gate blocks main submission (PLAN.md "If any gate fails: stop
expansion") by raising, rather than silently continuing past a failed check.
"""

from __future__ import annotations

import json

import pytest

import joint.gating.gate as gate


def _passing(name: str):
    return lambda config: {"pass": True, "name": name}


def _failing(name: str):
    return lambda config: {"pass": False, "name": name}


def test_run_gate_raises_when_any_gate_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "gate1_integrity", _passing("integrity"))
    monkeypatch.setattr(gate, "gate2_prior_mechanism", _passing("prior"))
    monkeypatch.setattr(gate, "gate3_support_mechanism", _passing("support"))
    monkeypatch.setattr(gate, "gate4_joint_rescue", _failing("rescue"))
    monkeypatch.setattr(gate, "gate5_robustness", _passing("robustness"))
    monkeypatch.setattr(gate, "gate6_precision", _passing("precision"))

    config = {"dataset": {"name": "bracs"}, "paths": {"outputs": str(tmp_path)}}
    with pytest.raises(RuntimeError, match="joint_rescue"):
        gate.run_gate(config)

    diagnostics = json.loads((tmp_path / "data" / "diagnostics.json").read_text())
    assert diagnostics["overall_pass"] is False
    assert diagnostics["gates"]["joint_rescue"]["pass"] is False


def test_run_gate_passes_when_every_gate_passes(tmp_path, monkeypatch):
    for gate_name in (
        "gate1_integrity",
        "gate2_prior_mechanism",
        "gate3_support_mechanism",
        "gate4_joint_rescue",
        "gate5_robustness",
        "gate6_precision",
    ):
        monkeypatch.setattr(gate, gate_name, _passing(gate_name))

    config = {"dataset": {"name": "bracs"}, "paths": {"outputs": str(tmp_path)}}
    result = gate.run_gate(config)
    assert result["overall_pass"] is True
