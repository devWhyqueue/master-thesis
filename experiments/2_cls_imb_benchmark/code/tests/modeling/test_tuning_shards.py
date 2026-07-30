from __future__ import annotations

from types import SimpleNamespace

import pytest

from imbalance_benchmark.modeling.workflows.tuning.tuning_reduction import (
    combined_cost,
    select_candidate_payload,
)
from imbalance_benchmark.modeling.workflows.tuning.candidate_registry import (
    register_candidates,
    write_round_grids,
)
from imbalance_benchmark.modeling.workflows.tuning.tuning_artifacts import (
    validate_shard_payload,
)
from imbalance_benchmark.modeling.workflows.tuning.tuning_shards import (
    ShardSpec,
    _observation_keys,
)
from imbalance_benchmark.modeling.workflows.tuning.tuning_schedule import (
    array_coordinates,
    bundled_array_size,
    bundled_observation_array_size,
    resolve_round_shard_spec,
    resolve_shard_spec,
)


def _payload(index: int, balanced_accuracy: float) -> dict[str, object]:
    return {
        "candidate_index": index,
        "config": {"lr": [0.1, 0.2][index]},
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


def test_shard_reduction_uses_frozen_candidate_order_for_exact_ties() -> None:
    selected = select_candidate_payload([_payload(1, 0.5), _payload(0, 0.5)])

    assert selected["config"] == {"lr": 0.1}


def test_shard_mapping_covers_frozen_candidates_once() -> None:
    grids = {"ce": [{"lr": 0.1}, {"lr": 0.2}], "weighted_ce": [{"lr": 0.3}]}
    specs = [
        resolve_shard_spec(index, "base", "natural", ("ce", "weighted_ce"), grids)
        for index in range(20)
    ]

    realized = [(spec.method, spec.candidate_index) for spec in specs if spec]
    assert realized == [("ce", 0), ("ce", 1), ("weighted_ce", 0)]


def test_resolve_round_shard_spec_addresses_only_new_configs(tmp_path) -> None:
    windows = {
        "ce": {"lr_window": [3e-4, 1e-3, 3e-3, 1e-2], "strength_window": None}
    }
    write_round_grids(tmp_path, "moderate", 1, windows)
    # 3e-4, 1e-3, 3e-3 already trained in round 0; only 1e-2 is new this round.
    register_candidates(tmp_path, "moderate", "ce", [{"lr": v} for v in windows["ce"]["lr_window"][:3]], round_index=0)

    spec = resolve_round_shard_spec(tmp_path, "moderate", 0, "base", ("ce",))
    assert spec == ShardSpec("moderate", "ce", 0, "base", round=1)
    assert resolve_round_shard_spec(tmp_path, "moderate", 1, "base", ("ce",)) is None


def test_resolve_round_shard_spec_skips_a_method_with_nothing_new(tmp_path) -> None:
    write_round_grids(
        tmp_path,
        "moderate",
        1,
        {
            "ce": {"lr_window": [3e-4, 1e-3, 3e-3, 1e-2], "strength_window": None},
            "weighted_ce": {"lr_window": [3e-4], "strength_window": [1.0]},
        },
    )
    register_candidates(tmp_path, "moderate", "weighted_ce", [{"lr": 3e-4, "parameter": 1.0}], round_index=0)

    # weighted_ce has nothing new: its whole fixed-slot range (0-15) is dead this
    # round; ce's one new candidate starts at its reserved offset, index 16.
    assert resolve_round_shard_spec(tmp_path, "moderate", 0, "base", ("weighted_ce", "ce")) is None
    spec = resolve_round_shard_spec(tmp_path, "moderate", 16, "base", ("weighted_ce", "ce"))
    assert spec.method == "ce"


def test_array_coordinates_cross_candidates_with_every_observation() -> None:
    assert [array_coordinates(index, None, 6) for index in range(8)] == [
        (0, 0),
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (0, 5),
        (1, 0),
        (1, 1),
    ]
    assert array_coordinates(0, None, 6, candidate_offset=1) == (1, 0)
    assert array_coordinates(23, None, 6, candidate_offset=1) == (4, 5)


def test_bundled_array_size_covers_every_shard_once() -> None:
    assert bundled_array_size(792, 8) == 99
    assert bundled_array_size(15, 8) == 2
    assert bundled_observation_array_size(132, 6, 4) == 198


def test_resume_rejects_a_stale_freeze_fingerprint() -> None:
    payload = {"fingerprint": ["old"], "complete": True}

    with pytest.raises(RuntimeError, match="fingerprint"):
        validate_shard_payload(payload, ["current"])


def test_shard_observations_cover_assignment_split_and_seed_once() -> None:
    scopes = [
        SimpleNamespace(assignment=assignment, split_index=split_index)
        for assignment in ("a", "b")
        for split_index in range(3)
    ]

    observations = _observation_keys(scopes, [11, 22])

    identities = {
        (item["assignment"], item["split_index"], item["seed"])
        for item in observations
    }
    assert len(observations) == len(identities) == 12


def test_resume_accepts_only_a_complete_matching_observation_set() -> None:
    spec = ShardSpec("natural", "ce", 0, "base")
    keys = [
        {
            "scope_index": 0,
            "assignment": "native",
            "split_index": 0,
            "seed_index": index,
            "seed": seed,
        }
        for index, seed in enumerate((11, 22))
    ]
    payload = {
        "complete": True,
        "fingerprint": ["a", "b", "c"],
        "spec": {
            "condition": "natural",
            "method": "ce",
            "candidate_index": 0,
            "phase": "base",
            "observation_index": None,
            "round": 0,
        },
        "seeds": [11, 22],
        "scope_count": 1,
        "observation_keys": keys,
        "cost_records": [{}, {}],
        "metrics": [
            {
                **key,
                "balanced_accuracy": 0.5,
                "macro_f1": 0.5,
                "nll": 0.5,
            }
            for key in reversed(keys)
        ],
    }

    validate_shard_payload(payload, ["a", "b", "c"], spec)
    payload["observation_keys"] = [keys[0], keys[0]]
    with pytest.raises(RuntimeError, match="missing or duplicated"):
        validate_shard_payload(payload, ["a", "b", "c"], spec)


def test_parallel_cost_sums_exposure_and_accelerator_time() -> None:
    def payload(start: float, end: float, seconds: float, memory: int) -> dict:
        return {
            "started_at": start,
            "completed_at": end,
            "accelerator_seconds": seconds,
            "peak_accelerator_memory_bytes": memory,
            "hardware": {"device": "gpu", "job_id": str(start)},
            "cost_records": [
                {
                    "processed_examples": 10,
                    "processed_instances": 0,
                    "unique_training_examples": 5,
                    "total_parameters": 4,
                    "trainable_parameters": 3,
                    "training_footprint_parameters": 4,
                }
            ],
        }

    cost = combined_cost([payload(10, 30, 20, 100), payload(20, 40, 20, 200)])

    assert cost["wall_clock_seconds"] == 30
    assert cost["accelerator_hours"] == pytest.approx(40 / 3600)
    assert cost["peak_accelerator_memory_bytes"] == 200
    assert cost["processed_examples"] == 20
