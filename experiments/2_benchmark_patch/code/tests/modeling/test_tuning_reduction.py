from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from imbalance_benchmark.modeling.workflows.tuning.aggregation.candidate_registry import (
    load_registry,
    registry_lookup,
)
from imbalance_benchmark.modeling.workflows.tuning.tuning_artifacts import (
    ShardSpec,
    shard_path,
    write_atomic,
)
from imbalance_benchmark.modeling.workflows.tuning.tuning_execution import (
    write_base_selection,
)
from imbalance_benchmark.modeling.workflows.tuning.tuning_reduction import (
    ReduceRound,
    combine_selection,
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


def test_reduce_phase_aliases_ces_metrics_as_a_free_parameter_zero_candidate(
    tmp_path: Path,
):
    ce_grid = [{"lr": 1e-4}, {"lr": 3e-4}]
    _write_candidate(tmp_path, ShardSpec("moderate", "ce", 0, "base"), ce_grid[0], 0.5)
    _write_candidate(tmp_path, ShardSpec("moderate", "ce", 1, "base"), ce_grid[1], 0.9)

    wce_grid = [
        {"parameter": p, "lr": lr} for p in (0.25, 0.5) for lr in (1e-4, 3e-4)
    ]
    for index, cfg in enumerate(wce_grid):
        _write_candidate(
            tmp_path, ShardSpec("moderate", "weighted_ce", index, "base"), cfg, 0.3
        )

    selections, payloads = reduce_phase(
        tmp_path,
        "moderate",
        "base",
        ("ce", "weighted_ce"),
        {"ce": ce_grid, "weighted_ce": wce_grid},
        ReduceRound(FINGERPRINT),
    )

    assert selections["weighted_ce"] == {"parameter": 0.0, "lr": 3e-4}
    # Aliased candidates never spend a shard; only the 2 CE + 4 weighted_ce
    # trained shards are ever counted.
    assert len(payloads) == 6


def test_reduce_phase_skips_ce_alias_when_ce_has_not_been_reduced_this_call(
    tmp_path: Path,
):
    wce_grid = [{"parameter": 0.25, "lr": 1e-4}, {"parameter": 0.5, "lr": 1e-4}]
    for index, cfg in enumerate(wce_grid):
        _write_candidate(
            tmp_path, ShardSpec("moderate", "weighted_ce", index, "base"), cfg, 0.3
        )

    selections, _ = reduce_phase(
        tmp_path,
        "moderate",
        "base",
        ("weighted_ce",),
        {"weighted_ce": wce_grid},
        ReduceRound(FINGERPRINT),
    )

    assert selections["weighted_ce"] == {"parameter": 0.25, "lr": 1e-4}


def test_ce_soft_hybrids_train_their_own_zero_strength_candidate_not_ces(
    tmp_path: Path,
):
    """Regression: ce_soft_f1/mcc train under a forced balanced sampler, so their
    weight=0 point is balanced-sampling CE, not CE - it must never be aliased
    from CE's metrics even when CE was reduced in the same call.
    """
    ce_grid = [{"lr": 1e-4}]
    _write_candidate(tmp_path, ShardSpec("moderate", "ce", 0, "base"), ce_grid[0], 0.9)

    soft_grid = [{"parameter": 0.0, "lr": 1e-4}, {"parameter": 0.25, "lr": 1e-4}]
    soft_scores = [0.8, 0.3]
    for index, (cfg, score) in enumerate(zip(soft_grid, soft_scores)):
        _write_candidate(
            tmp_path, ShardSpec("moderate", "ce_soft_f1", index, "base"), cfg, score
        )

    selections, payloads = reduce_phase(
        tmp_path,
        "moderate",
        "base",
        ("ce", "ce_soft_f1"),
        {"ce": ce_grid, "ce_soft_f1": soft_grid},
        ReduceRound(FINGERPRINT),
    )

    # 1 CE + 2 ce_soft_f1 candidates, all trained - no free CE-aliased entry added.
    assert len(payloads) == 3
    assert selections["ce_soft_f1"] == {"parameter": 0.0, "lr": 1e-4}


def test_write_base_selection_merges_instead_of_overwriting(tmp_path: Path):
    write_base_selection(tmp_path, "moderate", {"ce": {"lr": 1e-4}})
    write_base_selection(tmp_path, "moderate", {"weighted_ce": {"parameter": 0.5, "lr": 1e-4}})

    path = tmp_path / "tuning_shards" / "base_selections_moderate.json"
    saved = json.loads(path.read_text())

    assert saved == {
        "ce": {"lr": 1e-4},
        "weighted_ce": {"parameter": 0.5, "lr": 1e-4},
    }


def _payload(config: dict[str, Any], balanced_accuracy: float) -> dict[str, Any]:
    return {
        "config": config,
        "metrics": [
            {
                "scope_index": 0,
                "seed_index": 0,
                "seed": 11,
                "balanced_accuracy": balanced_accuracy,
                "macro_f1": 0.5,
                "nll": 0.5,
            }
        ],
    }


def test_focal_zero_gamma_is_not_aliased_from_cross_entropy():
    """focal's alpha applies at gamma=0, so its zero point is not plain CE."""
    grid = [{"lr": 1e-4, "parameter": 0.5}]
    candidates = [_payload({"lr": 1e-4, "parameter": 0.5}, 0.5)]
    ce_by_lr = {1e-4: _payload({"lr": 1e-4}, 0.9)}

    assert combine_selection("focal", grid, candidates, dict(ce_by_lr)) == {
        "lr": 1e-4,
        "parameter": 0.5,
    }
    # weighted_ce stays anchored: its parameter=0 really is plain CE.
    assert combine_selection("weighted_ce", grid, candidates, dict(ce_by_lr)) == {
        "lr": 1e-4,
        "parameter": 0.0,
    }
