from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from imbalance_benchmark.modeling.workflows.tuning.candidate_registry import (
    load_registry,
    registry_lookup,
)
from imbalance_benchmark.modeling.workflows.tuning.tuning_artifacts import (
    ShardSpec,
    shard_path,
    write_atomic,
)
from imbalance_benchmark.modeling.workflows.tuning.tuning_reduction import (
    ReduceRound,
    reduce_phase,
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


def test_reduce_phase_round_zero_matches_today(tmp_path: Path):
    grid = [{"lr": 1e-4}, {"lr": 3e-4}, {"lr": 1e-3}, {"lr": 3e-3}]
    for index, lr in enumerate([1e-4, 3e-4, 1e-3, 3e-3]):
        _write_candidate(
            tmp_path, ShardSpec("moderate", "ce", index, "base"), {"lr": lr}, 0.5
        )
    selections, payloads = reduce_phase(
        tmp_path, "moderate", "base", ("ce",), {"ce": grid}, ReduceRound(FINGERPRINT)
    )
    assert selections["ce"] == {"lr": 1e-4}
    assert len(payloads) == 4


def test_reduce_phase_reuses_a_round_zero_candidate_without_rewriting_it(tmp_path: Path):
    round0_grid = [{"lr": 1e-4}, {"lr": 3e-4}, {"lr": 1e-3}, {"lr": 3e-3}]
    for index, cfg in enumerate(round0_grid):
        _write_candidate(tmp_path, ShardSpec("moderate", "ce", index, "base"), cfg, 0.5)
    reduce_phase(
        tmp_path, "moderate", "base", ("ce",), {"ce": round0_grid}, ReduceRound(FINGERPRINT)
    )

    round1_grid = [{"lr": 3e-4}, {"lr": 1e-3}, {"lr": 3e-3}, {"lr": 1e-2}]
    # Only the genuinely new column (1e-2) is written this round.
    new_spec = ShardSpec("moderate", "ce", 0, "base", round=1)
    _write_candidate(tmp_path, new_spec, {"lr": 1e-2}, 0.9)

    selections, payloads = reduce_phase(
        tmp_path,
        "moderate",
        "base",
        ("ce",),
        {"ce": round1_grid},
        ReduceRound(FINGERPRINT, index=1),
    )

    assert selections["ce"] == {"lr": 1e-2}
    assert len(payloads) == 4
    registry = load_registry(tmp_path, "moderate")
    assert registry_lookup(registry, "ce", {"lr": 3e-4}) == (0, 1)
    assert registry_lookup(registry, "ce", {"lr": 1e-2}) == (1, 0)


def test_reduce_phase_raises_when_a_new_round_candidate_was_never_trained(
    tmp_path: Path,
):
    round0_grid = [{"lr": 1e-4}]
    _write_candidate(
        tmp_path, ShardSpec("moderate", "ce", 0, "base"), round0_grid[0], 0.5
    )
    reduce_phase(
        tmp_path, "moderate", "base", ("ce",), {"ce": round0_grid}, ReduceRound(FINGERPRINT)
    )

    with pytest.raises(RuntimeError, match="Missing tuning shard"):
        reduce_phase(
            tmp_path,
            "moderate",
            "base",
            ("ce",),
            {"ce": [{"lr": 1e-4}, {"lr": 3e-4}]},
            ReduceRound(FINGERPRINT, index=1),
        )
