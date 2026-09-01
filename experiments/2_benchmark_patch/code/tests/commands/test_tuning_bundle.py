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


def test_scale_thread_env_divides_task_cpus_across_workers(monkeypatch) -> None:
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "16")

    shard_workers._scale_thread_env(4)

    assert shard_workers.os.environ["OMP_NUM_THREADS"] == "4"


def test_scale_thread_env_never_rounds_below_one(monkeypatch) -> None:
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "3")

    shard_workers._scale_thread_env(4)

    assert shard_workers.os.environ["OMP_NUM_THREADS"] == "1"


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
    started: list[tuple] = []

    class FakeProcess:
        def __init__(self, target, args) -> None:
            started.append(args)
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

    assert sorted(spec for _run_one, batch, _margin in started for spec in batch) == [
        0,
        1,
        2,
        3,
    ]
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


_GIB = 1024**3


def test_vram_capped_workers_shrinks_ceiling_on_a_40gb_card(monkeypatch) -> None:
    monkeypatch.setattr(shard_workers.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        shard_workers.torch.cuda, "mem_get_info", lambda: (37 * _GIB, 40 * _GIB)
    )

    resolved = shard_workers._vram_capped_workers(
        parallel_fits=3, per_fit_bytes=19 * _GIB
    )

    assert resolved == 1


def test_vram_capped_workers_keeps_ceiling_on_an_80gb_card(monkeypatch) -> None:
    monkeypatch.setattr(shard_workers.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        shard_workers.torch.cuda, "mem_get_info", lambda: (74 * _GIB, 80 * _GIB)
    )

    resolved = shard_workers._vram_capped_workers(
        parallel_fits=3, per_fit_bytes=19 * _GIB
    )

    assert resolved == 3


def test_vram_capped_workers_is_unchanged_without_cuda(monkeypatch) -> None:
    monkeypatch.setattr(shard_workers.torch.cuda, "is_available", lambda: False)

    assert shard_workers._vram_capped_workers(parallel_fits=3, per_fit_bytes=19 * _GIB) == 3


def test_vram_capped_workers_is_unchanged_when_per_fit_bytes_unknown(monkeypatch) -> None:
    monkeypatch.setattr(shard_workers.torch.cuda, "is_available", lambda: True)

    assert shard_workers._vram_capped_workers(parallel_fits=3, per_fit_bytes=0) == 3


def test_run_guarded_stops_cleanly_before_an_overrunning_item(monkeypatch) -> None:
    monkeypatch.setenv("SLURM_JOB_END_TIME", str(shard_workers.time.time() + 60))
    calls: list[int] = []

    shard_workers._run_guarded(calls.append, [1, 2, 3], deadline_margin_seconds=3600)

    assert calls == []


def test_run_guarded_runs_every_item_when_margin_is_ample(monkeypatch) -> None:
    monkeypatch.delenv("SLURM_JOB_END_TIME", raising=False)
    calls: list[int] = []

    shard_workers._run_guarded(calls.append, [1, 2, 3], deadline_margin_seconds=10)

    assert calls == [1, 2, 3]


def test_max_bank_bytes_is_zero_for_fake_specs() -> None:
    assert tuning._max_bank_bytes({}, {}, [0, 1]) == 0


def test_max_bank_bytes_takes_the_worst_condition_and_assignment(monkeypatch) -> None:
    seen: list[tuple[str, str]] = []

    def fake_bank_bytes_for(base, condition, assignment, dtype):
        seen.append((condition, assignment))
        return {"balanced": 1, "severe": 2}[condition] * _GIB

    monkeypatch.setattr(tuning, "bank_bytes_for", fake_bank_bytes_for)
    freeze = {"tail_assignments": {"a": [], "b": []}}
    specs = [
        tuning.ShardSpec("balanced", "ce", 0, "base"),
        tuning.ShardSpec("severe", "ce", 0, "base"),
    ]

    result = tuning._max_bank_bytes({}, freeze, specs)

    assert result == 2 * _GIB + tuning._MEASURED_TRANSIENT_BYTES
    assert ("balanced", "native") in seen
    assert {("severe", "a"), ("severe", "b")}.issubset(set(seen))


def test_max_item_seconds_scales_with_expected_observations() -> None:
    freeze = {
        "tail_assignments": {"a": [], "b": [], "c": []},
        "seed_roles": {"tuning_initialization_0": 1, "tuning_initialization_1": 2},
    }
    balanced = tuning.ShardSpec("balanced", "ce", 0, "base")
    severe = tuning.ShardSpec("severe", "ce", 0, "base")

    only_balanced = tuning._max_item_seconds(freeze, [balanced])
    with_severe = tuning._max_item_seconds(freeze, [balanced, severe])

    assert only_balanced == tuning._MEASURED_PER_FIT_SECONDS * 6
    assert with_severe == tuning._MEASURED_PER_FIT_SECONDS * 18


def test_max_item_seconds_is_one_fit_for_a_bundled_observation() -> None:
    freeze = {"tail_assignments": {"native": []}}
    spec = tuning.ShardSpec("natural", "ce", 0, "base", observation_index=2)

    assert tuning._max_item_seconds(freeze, [spec]) == tuning._MEASURED_PER_FIT_SECONDS
