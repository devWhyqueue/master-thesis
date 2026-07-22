from __future__ import annotations

from argparse import Namespace

from imbalance_benchmark.modeling.workflows.tuning import tuning_bundle


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
    )

    assert tuning_bundle.run_shard_bundle(args)
    indices = [
        int(command[command.index("--shard-index") + 1]) for command in commands
    ]
    assert indices == list(range(24, 32))
    assert all("--shards-per-task" not in command for command in commands)
