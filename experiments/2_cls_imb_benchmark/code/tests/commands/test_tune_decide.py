from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from imbalance_benchmark.commands.tuning import decide
from imbalance_benchmark.modeling.context import GRIDS, LEARNING_RATE_GRID
from imbalance_benchmark.modeling.workflows.tuning.candidate_registry import (
    load_round_grids,
    load_round_state,
    merge_round_state,
    write_round_grids,
)
from imbalance_benchmark.modeling.workflows.tuning.search_windows import LR_ENVELOPE
from imbalance_benchmark.modeling.workflows.tuning.tuning_artifacts import (
    ShardSpec,
    shard_path,
    write_atomic,
)
from imbalance_benchmark.modeling.workflows.tuning.tuning_execution import (
    write_base_selection,
)
from imbalance_benchmark.modeling.workflows.tuning.tuning_rounds import decide_next_round


def _args(condition="moderate", phase="base", round_index=0) -> argparse.Namespace:
    return argparse.Namespace(config="config.yaml", condition=condition, phase=phase, round=round_index)


def test_this_round_windows_round_zero_uses_frozen_defaults():
    windows = decide._this_round_windows(Path("root"), "moderate", "base", 0, ("ce", "focal"), 9)
    assert windows["ce"] == (LEARNING_RATE_GRID, None)
    assert windows["focal"] == (LEARNING_RATE_GRID, GRIDS["focal"])


def test_round_zero_windows_keep_the_parameter_dimension_for_fixed_grid_methods():
    """oko/weighted_ce/balanced_sampling are not in STRENGTH_ENVELOPES (their
    grid is fixed, never adaptively shifted) but they still train a real
    "parameter" per candidate - a strength window of None here makes a
    later round's expand_grid drop that key and crash the method's fit
    function (KeyError: 'parameter'), which is exactly what happened the
    first time oko reached round 1 on real cluster data."""
    windows = decide._round0_windows(("oko", "weighted_ce", "balanced_sampling"), 9)
    assert windows["oko"] == (LEARNING_RATE_GRID, [float(v) for v in GRIDS["oko"]])
    assert windows["weighted_ce"][1] == [float(v) for v in GRIDS["weighted_ce"]]
    assert windows["balanced_sampling"][1] == [float(v) for v in GRIDS["balanced_sampling"]]


def test_round_zero_windows_cap_oko_k_by_n_classes():
    """OKO samples k distinct "odd classes" without replacement, so k must
    stay <= n_classes - 1 or sampling itself crashes (ValueError: Cannot
    take a larger sample than population) - this is what broke oko's
    round 1 on BRACS (7 classes) once its window carried the full [1,2,4,8]
    grid forward instead of get_grid_configs' frozen-round k <= n-1 cap."""
    windows = decide._round0_windows(("oko",), 7)
    assert windows["oko"] == (LEARNING_RATE_GRID, [1.0, 2.0, 4.0])


def test_this_round_windows_later_round_reads_signed_grids(tmp_path):
    from imbalance_benchmark.modeling.workflows.tuning.candidate_registry import (
        write_round_grids,
    )

    write_round_grids(
        tmp_path, "moderate", "base", 1, {"ce": {"lr_window": LR_ENVELOPE[3:7], "strength_window": None}}
    )
    windows = decide._this_round_windows(tmp_path, "moderate", "base", 1, ("ce",), 9)
    assert windows["ce"] == (LR_ENVELOPE[3:7], None)


def test_reduce_this_round_passes_expected_observations_to_reduce_phase(monkeypatch):
    """Natural-group base shards are written observation-bundled (no flat
    ``candidate=N.json``), so ``reduce_phase`` must get a real expected-
    observation count or ``load_candidate`` raises "Missing tuning shard"
    on every candidate - this is what broke every condition's round-0
    decide the first time it ran against real bundled shard output."""
    captured = {}

    def fake_reduce_phase(root, condition, phase, methods, grids, reduce_round, expected=None):
        captured["expected"] = expected
        return {}, []

    monkeypatch.setattr(decide, "reduce_phase", fake_reduce_phase)

    freeze = {
        "method_grids": {"ce": [{"lr": 1e-4}]},
        "seed_roles": {"tuning_initialization_0": 1, "tuning_initialization_1": 2},
        "tail_assignments": {"native": []},
    }
    decide._reduce_this_round(
        {"data": Path("root")}, freeze, "moderate", "base", 0, ("ce",), ["fp"]
    )

    assert captured["expected"] == 6  # len(assignments)=1 * 3 splits * 2 seeds


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


def test_advance_keeps_the_resolved_strength_window_when_only_lr_still_shifts(
    tmp_path, monkeypatch
):
    """When a method's strength axis resolves before its lr axis, the next
    round must still carry the resolved strength window forward (like lr's
    ``next_lr_window or windows[method][0]`` already does), not drop it to
    None - a dropped window here loses the "parameter" key entirely and
    silently trains the next round without that method's control applied,
    which is exactly what happened to ce_soft_f1's round 1 on the cluster."""
    monkeypatch.setattr(decide, "check_queue_cap", lambda: None)
    submitted = []

    def fake_submit(config, config_path, job):
        submitted.append(job)
        return f"id-{len(submitted)}"

    base = {"data": tmp_path}
    windows = {"focal": (LEARNING_RATE_GRID, GRIDS["focal"])}
    state = decide_next_round(
        "focal",
        {"lr": LEARNING_RATE_GRID[-1], "parameter": GRIDS["focal"][1]},
        LEARNING_RATE_GRID,
        GRIDS["focal"],
    )
    assert state.strength.resolved  # interior winner: strength axis is done

    decide._advance(
        base, {}, "config.yaml", _args(round_index=0), {"focal": state}, windows, False,
        submit=fake_submit,
    )

    grids = load_round_grids(tmp_path, "moderate", "base")
    assert grids["windows"]["focal"]["strength_window"] == GRIDS["focal"]


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


def test_cmd_tune_decide_handles_a_later_round_with_only_some_methods_active(
    tmp_path, monkeypatch
):
    """A method resolved in an earlier round must survive a later round's
    decide call for a *different* still-shifting method: no KeyError from
    the round's active-method subset, and no lost base selection."""
    base = {"data": tmp_path}
    merge_round_state(
        tmp_path, "moderate", {"ce": {"resolved": True, "tuning_limited": False}}
    )
    write_base_selection(tmp_path, "moderate", {"ce": {"lr": 1e-4}})
    write_round_grids(
        tmp_path,
        "moderate",
        "base",
        1,
        {"weighted_ce": {"lr_window": [1e-4, 3e-4], "strength_window": [0.25, 0.5]}},
    )
    grid = [{"parameter": p, "lr": lr} for p in (0.25, 0.5) for lr in (1e-4, 3e-4)]
    for index, cfg in enumerate(grid):
        write_atomic(
            shard_path(
                tmp_path, ShardSpec("moderate", "weighted_ce", index, "base", round=1)
            ),
            {
                "complete": True,
                "fingerprint": ["fp"],
                "spec": {
                    "condition": "moderate",
                    "method": "weighted_ce",
                    "candidate_index": index,
                    "phase": "base",
                    "observation_index": None,
                    "round": 1,
                },
                "seeds": [11],
                "scope_count": 1,
                "observation_keys": [{"scope_index": 0, "seed_index": 0, "seed": 11}],
                "cost_records": [{}],
                "config": cfg,
                "metrics": [
                    {
                        "scope_index": 0,
                        "seed_index": 0,
                        "seed": 11,
                        "balanced_accuracy": 0.5,
                        "macro_f1": 0.5,
                        "nll": 0.5,
                    }
                ],
            },
        )

    fake_regime = type("Regime", (), {"is_mil": False})()
    monkeypatch.setattr(
        decide,
        "_frozen_shard_context",
        lambda args: (
            base,
            [({"data": tmp_path}, fake_regime, None)],
            {
                "runtime_config": {},
                "seed_roles": {"tuning_initialization_0": 11},
                "class_names": ["a", "b", "c", "d", "e", "f", "g"],
            },
            ["fp"],
        ),
    )
    monkeypatch.setattr(decide, "_is_excluded", lambda paths: False)
    advanced = []
    monkeypatch.setattr(
        decide, "_advance", lambda *a: advanced.append(a)
    )

    decide.cmd_tune_decide(_args(round_index=1))

    state = load_round_state(tmp_path, "moderate")
    assert state["ce"] == {"resolved": True, "tuning_limited": False}
    assert "weighted_ce" in state

    saved = json.loads(
        (tmp_path / "tuning_shards" / "base_selections_moderate.json").read_text()
    )
    assert saved["ce"] == {"lr": 1e-4}
    assert "weighted_ce" in saved
    assert len(advanced) == 1


def _fake_job(name: str):
    from imbalance_benchmark.hydra.rendering import SlurmJob

    return SlurmJob(name, f"tune-shard --phase dependent", "gpu-2h", 1, 4)
