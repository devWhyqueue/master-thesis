"""Verdict labels and the joint rule at the margin boundary."""

from __future__ import annotations

import numpy as np
import pytest

from ts_nll.analyze import joint_verdict, verdict

MARGIN, ALPHA = 0.05, 0.05 / 32


def _dist(level: float) -> np.ndarray:
    return np.concatenate([[99.0], np.full(1000, level)])


@pytest.mark.parametrize(
    "level,decision",
    [(0.0, "no_meaningful_gain"), (-0.06, "method_better"), (-0.05, "inconclusive")],
)
def test_decisions(level: float, decision: str) -> None:
    result = verdict(_dist(level), MARGIN, ALPHA)
    assert result["decision"] == decision
    assert result["estimate"] == 99.0


def test_straddling_interval_is_inconclusive() -> None:
    rng = np.random.default_rng(0)
    dist = np.concatenate([[0.0], rng.normal(-0.05, 0.05, 10_000)])
    assert verdict(dist, MARGIN, ALPHA)["decision"] == "inconclusive"


def test_joint_needs_all() -> None:
    ok = {"a": "no_meaningful_gain", "b": "no_meaningful_gain"}
    assert joint_verdict(ok) == "confirmed"
    assert joint_verdict({**ok, "b": "inconclusive"}) == "not_confirmed"
