from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from imbalance_benchmark.commands.confirm import require_tuning_configs
from imbalance_benchmark.common import verify_signed_file, split_paths
from imbalance_benchmark.modeling.workflows.tuning.candidate_registry import (
    merge_round_state,
    register_candidates,
)
from imbalance_benchmark.modeling.workflows.tuning.tuning_artifacts import (
    ShardSpec,
    shard_path,
    write_atomic,
)
from imbalance_benchmark.modeling.workflows.tuning.tuning_execution import (
    _reduce_condition,
)

FINGERPRINT = ["fp"]
FREEZE = {
    "seed_roles": {"tuning_initialization_0": 11},
    "class_names": ["a", "b", "c", "d", "e", "f", "g"],
}


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
            "cost_records": [
                {
                    "processed_examples": 10,
                    "unique_training_examples": 5,
                    "total_parameters": 100,
                    "trainable_parameters": 100,
                    "training_footprint_parameters": 100,
                }
            ],
            "config": config,
            "metrics": [
                {
                    **keys[0],
                    "balanced_accuracy": balanced_accuracy,
                    "macro_f1": 0.5,
                    "nll": 0.5,
                }
            ],
            "started_at": 0.0,
            "completed_at": 1.0,
            "accelerator_seconds": 1.0,
            "peak_accelerator_memory_bytes": 1,
            "hardware": {"device": "cpu"},
        },
    )


def _seed_condition(root: Path, condition: str) -> None:
    """Register+write a resolved later-round CE and a tuning-limited edge crt."""
    round0 = [{"lr": 1e-4}, {"lr": 3e-4}, {"lr": 1e-3}, {"lr": 3e-3}]
    for index, cfg in enumerate(round0):
        _write_candidate(root, ShardSpec(condition, "ce", index, "base"), cfg, 0.5)
    register_candidates(root, condition, "ce", round0, round_index=0)
    register_candidates(root, condition, "ce", [{"lr": 1e-2}], round_index=1)
    _write_candidate(
        root, ShardSpec(condition, "ce", 0, "base", round=1), {"lr": 1e-2}, 0.9
    )

    crt_grid = [{"lr": 1e-4}, {"lr": 3e-4}, {"lr": 1e-3}, {"lr": 3e-3}]
    for index, cfg in enumerate(crt_grid):
        ba = 0.9 if cfg["lr"] == 3e-3 else 0.5
        _write_candidate(root, ShardSpec(condition, "crt", index, "dependent"), cfg, ba)
    register_candidates(root, condition, "crt", crt_grid, round_index=0)

    merge_round_state(
        root,
        condition,
        {
            "ce": {
                "resolved": True,
                "tuning_limited": False,
                "lr_window": [3e-4, 1e-3, 3e-3, 1e-2],
                "next_lr_window": None,
                "strength_window": None,
                "next_strength_window": None,
            },
            "crt": {
                "resolved": False,
                "tuning_limited": True,
                "lr_window": [1e-4, 3e-4, 1e-3, 3e-3],
                "next_lr_window": None,
                "strength_window": None,
                "next_strength_window": None,
            },
        },
    )


def test_reduce_condition_selects_the_terminal_round_winner_and_signs_it(tmp_path: Path):
    """Regression: this used to always re-select among round 0's frozen grid,
    so a method that resolved in a later round (like ce's shift to 1e-2 here)
    could never appear in the signed final selection confirmation reads."""
    base = {"root": tmp_path, "data": tmp_path / "data"}
    _seed_condition(base["data"], "moderate")

    _reduce_condition(base, FREEZE, FINGERPRINT, ("ce",), ("crt",), ("native",), "moderate")

    path = split_paths(base, 0)["data"] / "tuning_selections_moderate.json"
    verify_signed_file(path)
    selections = json.loads(path.read_text())["native"]["moderate"]

    assert selections["ce"] == {"lr": 1e-2}  # later-round winner, not round 0's best
    assert selections["crt"] == {"lr": 3e-3}  # tuning-limited edge winner still chosen

    confirmed = require_tuning_configs(base["data"], "moderate", selections, ("ce", "crt"))
    assert confirmed == selections  # confirmation consumes it unchanged


def test_reduce_condition_writes_the_same_signed_selection_to_every_split(tmp_path: Path):
    base = {"root": tmp_path, "data": tmp_path / "data"}
    _seed_condition(base["data"], "moderate")

    _reduce_condition(base, FREEZE, FINGERPRINT, ("ce",), ("crt",), ("native",), "moderate")

    for index in range(3):
        path = split_paths(base, index)["data"] / "tuning_selections_moderate.json"
        verify_signed_file(path)


def test_reduce_condition_cost_counts_every_registered_candidate_once(tmp_path: Path):
    base = {"root": tmp_path, "data": tmp_path / "data"}
    _seed_condition(base["data"], "moderate")

    _reduce_condition(base, FREEZE, FINGERPRINT, ("ce",), ("crt",), ("native",), "moderate")

    cost = json.loads(
        (split_paths(base, 0)["data"] / "tuning_search_cost_moderate.json").read_text()
    )
    # ce: 4 round-0 + 1 outward probe; crt: 4 round-0. 9 shards x 10 processed examples.
    assert cost["processed_examples"] == 90


def test_reduce_condition_aborts_when_tuning_is_not_locked(tmp_path: Path):
    base = {"root": tmp_path, "data": tmp_path / "data"}
    # No round-state file at all: the tuning lock was never granted.
    with pytest.raises(RuntimeError, match="not locked"):
        _reduce_condition(base, FREEZE, FINGERPRINT, ("ce",), ("crt",), ("native",), "moderate")
