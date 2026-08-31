from __future__ import annotations

from argparse import Namespace

from imbalance_benchmark.commands.tuning import shard as tuning
from imbalance_benchmark.commands.tuning import shard_workers
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
        tuning,
        "_run_shards",
        lambda received, indices: calls.append((received, indices)),
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
    """Patch candidates now construct scope-local banks, sequentially."""
    calls: list[int] = []
    context = (
        {"data": object()},
        [],
        {"runtime_config": {"dataset": {}}, "method_grids": {}},
        ["freeze"],
        [{"freeze"}],
    )
    monkeypatch.setattr(tuning, "_frozen_shard_context", lambda *_: context)
    monkeypatch.setattr(tuning, "_split_paths", lambda *_: [])
    monkeypatch.setattr(
        tuning,
        "_run_scope_local_shard",
        lambda *args: calls.append(args[-1].candidate_index),
    )
    monkeypatch.setattr(
        tuning,
        "requested_shard",
        lambda index, *_: tuning.ShardSpec("natural", "ce", index, "base"),
    )
    args = Namespace(
        config="config.yaml",
        phase="base",
        group="natural",
        shard_index=3,
        observation_index=None,
        observations_per_candidate=1,
        shard_offset=0,
        shards_per_task=2,
        bundle_by_observation=False,
    )

    tuning._run_shards(args, [6, 7])

    assert calls == [6, 7]


def test_single_shard_runs_without_bundle_indices(monkeypatch) -> None:
    args = Namespace(
        config="config.yaml",
        phase="base",
        group="natural",
        shard_index=0,
        observation_index=None,
        observations_per_candidate=1,
        shard_offset=0,
        shards_per_task=1,
        bundle_by_observation=False,
    )
    calls: list[list[int]] = []
    monkeypatch.setattr(tuning, "_run_shards", lambda _, indices: calls.append(indices))

    tuning.cmd_tune_shard(args)

    assert calls == [[0]]


def test_chunk_splits_round_robin_and_drops_empty_workers() -> None:
    assert shard_workers._chunk([0, 1, 2, 3, 4], 2) == [[0, 2, 4], [1, 3]]
    assert shard_workers._chunk([0], 4) == [[0]]


def test_execute_shards_runs_sequentially_when_parallel_fits_is_one(
    monkeypatch,
) -> None:
    calls: list[int] = []
    monkeypatch.setattr(
        tuning,
        "_run_scope_local_shard",
        lambda _args, _base, _freeze, _fp, _accepted, spec: calls.append(spec),
    )

    tuning._execute_shards(
        {}, Namespace(), {}, [], [], [10, 11], lambda index: index, parallel_fits=1
    )

    assert calls == [10, 11]


def test_execute_shards_packs_specs_across_child_processes(monkeypatch) -> None:
    started: list[list[int]] = []

    class FakeProcess:
        def __init__(self, target, args) -> None:
            started.append(args[-1])
            self.exitcode = 0

        def start(self) -> None:
            pass

        def join(self) -> None:
            pass

    class FakeContext:
        def Process(self, target, args) -> FakeProcess:
            return FakeProcess(target, args)

    monkeypatch.setattr(
        shard_workers.multiprocessing, "get_context", lambda *_a: FakeContext()
    )
    monkeypatch.setattr(shard_workers.time, "sleep", lambda _seconds: None)

    tuning._execute_shards(
        {}, Namespace(), {}, [], [], [0, 1, 2, 3], lambda index: index, parallel_fits=2
    )

    assert sorted(spec for batch in started for spec in batch) == [0, 1, 2, 3]
    assert len(started) == 2


def test_execute_shards_worker_failure_reaches_caller(monkeypatch) -> None:
    class FailedProcess:
        exitcode = 1

        def start(self) -> None:
            pass

        def join(self) -> None:
            pass

    class FailedContext:
        def Process(self, **_kwargs: object) -> FailedProcess:
            return FailedProcess()

    monkeypatch.setattr(
        shard_workers.multiprocessing, "get_context", lambda *_a: FailedContext()
    )
    monkeypatch.setattr(shard_workers.time, "sleep", lambda _seconds: None)

    try:
        tuning._execute_shards(
            {}, Namespace(), {}, [], [], [0, 1], lambda index: index, parallel_fits=2
        )
    except RuntimeError as error:
        assert "workers failed" in str(error)
    else:
        raise AssertionError("Expected a worker failure to raise RuntimeError.")
