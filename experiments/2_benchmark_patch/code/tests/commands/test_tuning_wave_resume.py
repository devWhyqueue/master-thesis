from __future__ import annotations

from pathlib import Path

import pytest

from imbalance_benchmark.commands.tuning import wave
from imbalance_benchmark.common import sign_file, write_json
from imbalance_benchmark.modeling.workflows.tuning.aggregation.candidate_registry import (
    round_state_path,
)


def _mark_decided(data: Path, condition: str) -> None:
    path = round_state_path(data, condition)
    write_json(path, {"ce": {"resolved": True, "tuning_limited": False}})
    sign_file(path)


def test_submit_terminal_skips_round_zero_decide_for_an_already_decided_condition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: --resume-tuning used to always resubmit every condition's
    round-0 decide once base shards looked complete, even for a condition
    that had already advanced past round 0 - recomputing its decision from
    round 0's frozen shards and clobbering the further-along state a later
    round already locked in, corrupting candidate-registry indexing."""
    base = {"root": tmp_path, "data": tmp_path / "data"}
    base["data"].mkdir(parents=True)
    _mark_decided(base["data"], "balanced")

    monkeypatch.setattr(wave, "_squeue_count", lambda: 0)
    monkeypatch.setattr(wave, "render_sbatch", lambda job, config, path: job.name)
    submitted: list[str] = []
    monkeypatch.setattr(
        wave, "_submit_script", lambda script, dry: submitted.append(script) or "1"
    )

    wave._submit_terminal({}, "/config.yaml", base)

    assert "tune-decide-base-balanced" not in submitted
    assert "tune-decide-base-natural" in submitted
    assert "tune-base-reduce" in submitted
