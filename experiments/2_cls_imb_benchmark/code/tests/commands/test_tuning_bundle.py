from __future__ import annotations

from argparse import Namespace

from imbalance_benchmark.modeling.workflows.tuning import tuning_bundle
from imbalance_benchmark.modeling.workflows.tuning.tuning_bundle import _bundle_indices


def test_bundle_launches_each_shard_once_without_recursive_bundling(monkeypatch) -> None:
    commands: list[list[str]] = []

    class Process:
        def __init__(self, command: list[str]) -> None:
            commands.append(command)

        def wait(self) -> int:
            return 0

    monkeypatch.setattr(tuning_bundle.subprocess, "Popen", Process)
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

    assert tuning_bundle.run_shard_bundle(args)
    indices = [
        int(command[command.index("--shard-index") + 1]) for command in commands
    ]
    assert indices == list(range(24, 32))
    assert all("--shards-per-task" not in command for command in commands)


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
