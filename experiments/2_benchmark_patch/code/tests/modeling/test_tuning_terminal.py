from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from imbalance_benchmark.modeling.context import GRIDS, LEARNING_RATE_GRID
from imbalance_benchmark.modeling.workflows.tuning.candidate_registry import (
    register_candidates,
    resolve_terminal_specs,
    terminal_cost_payloads,
)
from imbalance_benchmark.modeling.workflows.tuning.tuning_artifacts import (
    ShardSpec,
    shard_path,
    write_atomic,
)
from imbalance_benchmark.modeling.workflows.tuning.tuning_reduction import (
    reduce_terminal_phase,
    terminal_active_grids,
)

FINGERPRINT = ["fp"]


def _write_candidate(
    root: Path, spec: ShardSpec, config: dict[str, Any], balanced_accuracy: float
) -> None:
    keys = [{"scope_index": 0, "seed_index": 0, "seed": 11}]
    write_atomic(
        shard_path(root, spec),
        {
            "complete": True,
            "fingerprint": FINGERPRINT,
            "spec": {
                "condition": spec.condition,
                "method": spec.method,
                "candidate_index": spec.candidate_index,
                "phase": spec.phase,
                "observation_index": None,
                "round": spec.round,
            },
            "seeds": [11],
            "scope_count": 1,
            "observation_keys": keys,
            "cost_records": [{}],
            "config": config,
            "metrics": [
                {
                    **keys[0],
                    "balanced_accuracy": balanced_accuracy,
                    "macro_f1": 0.5,
                    "nll": 0.5,
                }
            ],
        },
    )


def test_terminal_active_grids_expands_each_methods_own_signed_window():
    """The terminal grid must come from the round-state window, not any frozen
    round-0 grid - a resolved method's winner can live several rounds out."""
    state = {
        "ce": {"lr_window": [3e-4, 1e-3, 3e-3, 1e-2], "strength_window": None},
        "focal": {
            "lr_window": LEARNING_RATE_GRID,
            "strength_window": [0.0, 0.5, 1.0, 1.5],
        },
    }
    grids = terminal_active_grids(state, ("ce", "focal"), 9)
    assert grids["ce"] == [{"lr": lr} for lr in [3e-4, 1e-3, 3e-3, 1e-2]]
    assert grids["focal"] == [
        {"parameter": p, "lr": lr}
        for p in [0.0, 0.5, 1.0, 1.5]
        for lr in LEARNING_RATE_GRID
    ]


def test_terminal_active_grids_omits_methods_absent_from_state():
    assert terminal_active_grids({}, ("post_hoc_logit_adjustment",), 9) == {}


def test_terminal_active_grids_falls_back_to_the_full_grid_for_fixed_grid_methods():
    """Regression: found live on the cluster. balanced_sampling/weighted_ce/oko
    never get a strength_window recorded in tuning_round_state (only the
    audited-unbounded controls adaptively shift one), so a naive expand_grid
    against the state dropped the parameter key entirely and every terminal
    candidate lookup failed with "not in registry" - the full frozen grid
    must be used instead, still crossed with the state's (possibly shifted)
    lr_window."""
    state = {
        "balanced_sampling": {
            "lr_window": [3e-5, 1e-4, 3e-4, 1e-3],
            "strength_window": None,
        },
        "oko": {"lr_window": LEARNING_RATE_GRID, "strength_window": None},
    }
    grids = terminal_active_grids(state, ("balanced_sampling", "oko"), 7)
    assert grids["balanced_sampling"] == [
        {"parameter": p, "lr": lr}
        for p in GRIDS["balanced_sampling"]
        for lr in [3e-5, 1e-4, 3e-4, 1e-3]
    ]
    # oko is additionally capped at n_classes - 1 = 6, dropping the 8.
    assert {cfg["parameter"] for cfg in grids["oko"]} == {1.0, 2.0, 4.0}


def test_resolve_terminal_specs_finds_a_later_round_candidate_without_registering_it(
    tmp_path: Path,
):
    round0 = [{"lr": 1e-4}, {"lr": 3e-4}, {"lr": 1e-3}, {"lr": 3e-3}]
    register_candidates(tmp_path, "moderate", "ce", round0, round_index=0)
    register_candidates(tmp_path, "moderate", "ce", [{"lr": 1e-2}], round_index=1)

    terminal_grid = [{"lr": 3e-4}, {"lr": 1e-3}, {"lr": 3e-3}, {"lr": 1e-2}]
    specs = resolve_terminal_specs(tmp_path, "moderate", "base", "ce", terminal_grid)

    assert [(spec.round, spec.candidate_index) for spec in specs] == [
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 0),
    ]


def test_resolve_terminal_specs_finds_an_int_registered_candidate_via_a_float_query(
    tmp_path: Path,
):
    """Load the integer-form key persisted before registry keys were normalized."""
    write_atomic(
        tmp_path / "tuning_shards" / "candidate_registry_moderate.json",
        {"oko|1|0.0001": {"round": 0, "candidate_index": 0}},
    )

    specs = resolve_terminal_specs(
        tmp_path, "moderate", "base", "oko", [{"parameter": 1.0, "lr": 1e-4}]
    )

    assert [(spec.round, spec.candidate_index) for spec in specs] == [(0, 0)]


def test_registry_normalization_prefers_the_earliest_trained_candidate_location(
    tmp_path: Path,
) -> None:
    write_atomic(
        tmp_path / "tuning_shards" / "candidate_registry_moderate.json",
        {
            "oko|1|0.0001": {"round": 0, "candidate_index": 0},
            "oko|1.0|0.0001": {"round": 1, "candidate_index": 5},
        },
    )

    specs = resolve_terminal_specs(
        tmp_path, "moderate", "base", "oko", [{"parameter": 1.0, "lr": 1e-4}]
    )

    assert specs == [ShardSpec("moderate", "oko", 0, "base", round=0)]


def test_resolve_terminal_specs_aborts_on_a_config_missing_from_the_registry(
    tmp_path: Path,
):
    """A round-state window naming a config the registry never recorded means
    the tuning lock was granted against stale or corrupted state - final
    reduction must refuse to guess, not silently retrain or skip it."""
    register_candidates(tmp_path, "moderate", "ce", [{"lr": 1e-4}], round_index=0)

    with pytest.raises(RuntimeError, match="Unregistered terminal candidate"):
        resolve_terminal_specs(tmp_path, "moderate", "base", "ce", [{"lr": 3e-4}])


def test_reduce_terminal_phase_selects_a_later_round_winner_via_the_registry(
    tmp_path: Path,
):
    """The frozen round-0 grid never contains a shifted window's new value, so
    only a registry-resolved terminal grid can surface it as the winner -
    this is exactly the bug: final reduction used to hardcode round 0."""
    round0 = [{"lr": 1e-4}, {"lr": 3e-4}, {"lr": 1e-3}, {"lr": 3e-3}]
    for index, cfg in enumerate(round0):
        _write_candidate(tmp_path, ShardSpec("moderate", "ce", index, "base"), cfg, 0.5)
    register_candidates(tmp_path, "moderate", "ce", round0, round_index=0)

    register_candidates(tmp_path, "moderate", "ce", [{"lr": 1e-2}], round_index=1)
    _write_candidate(
        tmp_path, ShardSpec("moderate", "ce", 0, "base", round=1), {"lr": 1e-2}, 0.9
    )

    terminal_grids = {"ce": [{"lr": 3e-4}, {"lr": 1e-3}, {"lr": 3e-3}, {"lr": 1e-2}]}
    selections, payloads = reduce_terminal_phase(
        tmp_path, "moderate", "base", ("ce",), terminal_grids, FINGERPRINT
    )

    assert selections["ce"] == {"lr": 1e-2}
    assert len(payloads) == 4  # only the terminal window's own 4 candidates


def test_reduce_terminal_phase_aborts_on_a_stale_shard_that_no_longer_matches_the_registry(
    tmp_path: Path,
):
    register_candidates(tmp_path, "moderate", "ce", [{"lr": 1e-4}], round_index=0)
    # The shard on disk disagrees with what the registry says lives here.
    _write_candidate(
        tmp_path, ShardSpec("moderate", "ce", 0, "base"), {"lr": 9e-4}, 0.5
    )

    with pytest.raises(RuntimeError, match="config mismatch"):
        reduce_terminal_phase(
            tmp_path, "moderate", "base", ("ce",), {"ce": [{"lr": 1e-4}]}, FINGERPRINT
        )


def test_terminal_cost_payloads_counts_round_zero_and_outward_probes_exactly_once(
    tmp_path: Path,
):
    """Realized tuning cost must cover every value the adaptive search ever
    trained, not just the terminal window - three of round 0's four
    candidates are no longer in the terminal window at all, but they still
    cost accelerator time and must still be counted, exactly once each."""
    round0 = [{"lr": 1e-4}, {"lr": 3e-4}, {"lr": 1e-3}, {"lr": 3e-3}]
    for index, cfg in enumerate(round0):
        _write_candidate(tmp_path, ShardSpec("moderate", "ce", index, "base"), cfg, 0.5)
    register_candidates(tmp_path, "moderate", "ce", round0, round_index=0)

    register_candidates(tmp_path, "moderate", "ce", [{"lr": 1e-2}], round_index=1)
    _write_candidate(
        tmp_path, ShardSpec("moderate", "ce", 0, "base", round=1), {"lr": 1e-2}, 0.9
    )

    payloads = terminal_cost_payloads(
        tmp_path, "moderate", "base", ("ce",), FINGERPRINT
    )

    assert len(payloads) == 5  # 4 from round 0 plus the 1 outward probe
    assert {p["config"]["lr"] for p in payloads} == {1e-4, 3e-4, 1e-3, 3e-3, 1e-2}
