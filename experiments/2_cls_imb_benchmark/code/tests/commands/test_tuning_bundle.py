from __future__ import annotations

from argparse import Namespace

from imbalance_benchmark.commands import tuning
from imbalance_benchmark.modeling.workflows.tuning.tuning_execution import (
    _bundle_indices,
)


def test_patch_bundle_runs_exact_shards_sequentially(monkeypatch) -> None:
    calls: list[tuple[Namespace, list[int]]] = []
    args = Namespace(
        config="config.yaml",
        phase="base",
        group="natural",
        shard_index=3,
        observation_index=None,
        observations_per_candidate=6,
        shard_offset=0,
        shards_per_task=8,
        bundle_by_observation=False,
    )
    monkeypatch.setattr(
        tuning, "_run_shards", lambda received, indices: calls.append((received, indices))
    )

    tuning.cmd_tune_shard(args)

    assert calls == [(args, list(range(24, 32)))]


def test_observation_bundles_keep_split_and_seed_homogeneous() -> None:
    assert _bundle_indices(3, 8, 6, by_observation=True) == [
        3,
        9,
        15,
        21,
        27,
        33,
        39,
        45,
    ]


def test_wsi_bundle_reuses_one_context_for_two_exact_shards(monkeypatch) -> None:
    """MIL candidates share one process-local feature cache, sequentially."""
    calls: list[int] = []
    context = (
        object(),
        [(object(), type("Regime", (), {"is_mil": True})(), object())],
        {"method_grids": {}},
        ["freeze"],
    )
    monkeypatch.setattr(tuning, "_frozen_shard_context", lambda *_: context)
    monkeypatch.setattr(tuning, "_is_excluded", lambda *_: False)
    monkeypatch.setattr(
        tuning, "_run_shard", lambda *args: calls.append(args[-1].candidate_index)
    )
    monkeypatch.setattr(
        tuning,
        "requested_shard",
        lambda index, *_: tuning.ShardSpec("natural", "ce", index, "base"),
    )
    args = Namespace(
        config="config.yaml", phase="base", group="natural", shard_index=3,
        observation_index=None, observations_per_candidate=1, shard_offset=0,
        shards_per_task=2, bundle_by_observation=False,
    )

    tuning._run_shards(args, [6, 7])

    assert calls == [6, 7]


def test_single_shard_runs_without_bundle_indices(monkeypatch) -> None:
    args = Namespace(
        config="config.yaml", phase="base", group="natural", shard_index=0,
        observation_index=None, observations_per_candidate=1, shard_offset=0,
        shards_per_task=1, bundle_by_observation=False,
    )
    calls: list[list[int]] = []
    monkeypatch.setattr(tuning, "_run_shards", lambda _, indices: calls.append(indices))

    tuning.cmd_tune_shard(args)

    assert calls == [[0]]
