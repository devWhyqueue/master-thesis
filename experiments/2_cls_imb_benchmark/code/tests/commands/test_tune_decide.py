from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from imbalance_benchmark.commands.tuning import decide
from imbalance_benchmark.modeling.context import GRIDS, LEARNING_RATE_GRID
from imbalance_benchmark.modeling.workflows.tuning.candidate_registry import (
    load_round_grids,
    load_round_state,
    merge_round_state,
)
from imbalance_benchmark.modeling.workflows.tuning.search_windows import LR_ENVELOPE
from imbalance_benchmark.modeling.workflows.tuning.tuning_rounds import decide_next_round


def _args(condition="moderate", phase="base", round_index=0) -> argparse.Namespace:
    return argparse.Namespace(config="config.yaml", condition=condition, phase=phase, round=round_index)


def test_this_round_windows_round_zero_uses_frozen_defaults():
    windows = decide._this_round_windows(Path("root"), "moderate", "base", 0, ("ce", "focal"))
    assert windows["ce"] == (LEARNING_RATE_GRID, None)
    assert windows["focal"] == (LEARNING_RATE_GRID, GRIDS["focal"])


def test_this_round_windows_later_round_reads_signed_grids(tmp_path):
    from imbalance_benchmark.modeling.workflows.tuning.candidate_registry import (
        write_round_grids,
    )

    write_round_grids(
        tmp_path, "moderate", "base", 1, {"ce": {"lr_window": LR_ENVELOPE[3:7], "strength_window": None}}
    )
    windows = decide._this_round_windows(tmp_path, "moderate", "base", 1, ("ce",))
    assert windows["ce"] == (LR_ENVELOPE[3:7], None)


def test_dependent_phase_started_false_when_no_state_file(tmp_path):
    assert decide._dependent_phase_started(tmp_path, "moderate") is False


def test_dependent_phase_started_false_before_crt_or_posthoc_appear(tmp_path):
    merge_round_state(tmp_path, "moderate", {"ce": {"resolved": True, "tuning_limited": False}})
    assert decide._dependent_phase_started(tmp_path, "moderate") is False


def test_dependent_phase_started_true_once_crt_appears(tmp_path):
    merge_round_state(tmp_path, "moderate", {"crt": {"resolved": False, "tuning_limited": False}})
    assert decide._dependent_phase_started(tmp_path, "moderate") is True


def test_advance_submits_next_round_when_a_method_is_still_shifting(tmp_path, monkeypatch):
    monkeypatch.setattr(decide, "check_queue_cap", lambda: None)
    submitted = []

    def fake_submit(config, config_path, job):
        submitted.append(job)
        return f"id-{len(submitted)}"

    base = {"data": tmp_path}
    windows = {"ce": (LEARNING_RATE_GRID, None)}
    edge_state = decide_next_round("ce", {"lr": LEARNING_RATE_GRID[-1]}, LEARNING_RATE_GRID)
    states = {"ce": edge_state}

    decide._advance(base, {}, "config.yaml", _args(round_index=0), states, windows, False, submit=fake_submit)

    assert len(submitted) == 2  # the round-1 shard array, then its decide job
    assert submitted[0].array_size > 0
    assert "--round 1" in submitted[0].command
    assert submitted[1].dependencies == ("id-1",)
    assert "tune-decide --phase base --condition moderate --round 1" in submitted[1].command
    grids = load_round_grids(tmp_path, "moderate", "base")
    assert grids["round"] == 1
    assert grids["windows"]["ce"]["lr_window"] == LR_ENVELOPE[3:7]


def test_advance_does_nothing_further_once_dependent_phase_resolves(tmp_path, monkeypatch):
    monkeypatch.setattr(decide, "check_queue_cap", lambda: None)
    submitted = []
    resolved = decide_next_round("crt", {"lr": LEARNING_RATE_GRID[1]}, LEARNING_RATE_GRID)

    decide._advance(
        {"data": tmp_path}, {}, "config.yaml", _args(phase="dependent"), {"crt": resolved},
        {"crt": (LEARNING_RATE_GRID, None)}, False,
        submit=lambda *a: submitted.append(a) or "id",
    )

    assert submitted == []


def test_advance_starts_dependent_phase_once_ce_resolves(tmp_path, monkeypatch):
    monkeypatch.setattr(decide, "check_queue_cap", lambda: None)
    monkeypatch.setattr(
        decide,
        "dependent_round_zero_jobs",
        lambda config, is_mil: [
            _fake_job("tune-dependent-posthoc-natural"),
            _fake_job("tune-dependent-crt-natural"),
            _fake_job("tune-dependent-controlled"),
        ],
    )
    submitted = []

    def fake_submit(config, config_path, job):
        submitted.append(job)
        return f"id-{len(submitted)}"

    ce_resolved = decide_next_round("ce", {"lr": LEARNING_RATE_GRID[1]}, LEARNING_RATE_GRID)

    decide._advance(
        {"data": tmp_path}, {}, "config.yaml", _args(phase="base"), {"ce": ce_resolved},
        {"ce": (LEARNING_RATE_GRID, None)}, False, submit=fake_submit,
    )

    names = [job.name for job in submitted]
    assert names == [
        "tune-dependent-posthoc-natural",
        "tune-dependent-crt-natural",
        "tune-dependent-controlled",
        "tune-decide-moderate-dependent-r0",
    ]
    assert submitted[-1].dependencies == ("id-1", "id-2", "id-3")


def test_advance_skips_dependent_phase_if_already_started(tmp_path, monkeypatch):
    monkeypatch.setattr(decide, "check_queue_cap", lambda: None)
    merge_round_state(tmp_path, "moderate", {"crt": {"resolved": False, "tuning_limited": False}})
    submitted = []
    ce_resolved = decide_next_round("ce", {"lr": LEARNING_RATE_GRID[1]}, LEARNING_RATE_GRID)

    decide._advance(
        {"data": tmp_path}, {}, "config.yaml", _args(phase="base"), {"ce": ce_resolved},
        {"ce": (LEARNING_RATE_GRID, None)}, False,
        submit=lambda *a: submitted.append(a) or "id",
    )

    assert submitted == []


def _fake_job(name: str):
    from imbalance_benchmark.hydra.rendering import SlurmJob

    return SlurmJob(name, f"tune-shard --phase dependent", "gpu-2h", 1, 4)
