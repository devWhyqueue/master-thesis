from __future__ import annotations

from imbalance_benchmark.hydra import resume
from imbalance_benchmark.modeling.workflows.tuning.tuning_artifacts import ShardSpec


def test_controlled_bundle_asks_expected_observations_for_the_specs_own_condition(
    monkeypatch,
) -> None:
    """Regression: resume.py used to pass the literal string "controlled" to
    expected_observations() instead of the resolved spec's own condition,
    always yielding 18 -- wrong for 'balanced' (6)."""
    captured: list[int | None] = []

    def fake_requested_shard(shard_index, phase, group, is_mil, grids, observation):
        condition = "balanced" if shard_index == 0 else "severe"
        return ShardSpec(condition, "ce", 0, phase)

    def fake_load_candidate(root, spec, fingerprint, expected, accepted):
        captured.append(expected)
        return {}

    monkeypatch.setattr(resume, "requested_shard", fake_requested_shard)
    monkeypatch.setattr(resume, "load_candidate", fake_load_candidate)

    freeze = {
        "method_grids": {},
        "tail_assignments": {"a": [], "b": [], "c": []},
        "seed_roles": {"tuning_initialization_0": 1, "tuning_initialization_1": 2},
    }

    resume._controlled_bundle_complete(
        task_index=0,
        bundle_size=2,
        total=2,
        is_mil=False,
        freeze=freeze,
        fingerprint=[],
        assignments=("a", "b", "c"),
        base={"data": object()},
        accepted=[],
    )

    assert captured == [6, 18]
