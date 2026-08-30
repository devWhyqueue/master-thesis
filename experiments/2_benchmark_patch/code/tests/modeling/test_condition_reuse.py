from __future__ import annotations

from pathlib import Path

from imbalance_benchmark.common import sign_file, split_paths, write_json
from imbalance_benchmark.modeling.workflows.tuning.tuning_schedule import (
    condition_is_reusable,
)


def _seed_condition_outputs(
    base: dict[str, Path], fingerprint: list[str], with_fingerprint: bool = True
) -> None:
    selections = {"native": {"natural": {"ce": {"lr": 1e-4}}}}
    cost = {"wall_clock_seconds": 1.0, "accelerator_hours": 1.0, "processed_examples": 1}
    if with_fingerprint:
        cost["fingerprint"] = fingerprint
    for index in range(3):
        data = split_paths(base, index)["data"]
        selection_path = data / "tuning_selections_natural.json"
        write_json(selection_path, selections)
        sign_file(selection_path)
        write_json(data / "tuning_search_cost_natural.json", cost)


def test_condition_is_reusable_true_when_the_freeze_fingerprint_matches(tmp_path):
    base = {"root": tmp_path, "data": tmp_path / "data"}
    _seed_condition_outputs(base, ["a", "b", "c"])
    assert condition_is_reusable(base, "natural", ("ce",), ("native",), ["a", "b", "c"])


def test_condition_is_reusable_false_when_the_freeze_fingerprint_is_stale(tmp_path):
    """Regression: a pre-plan-07 selection reused forever regardless of a
    later freeze amendment, since reuse never checked freeze provenance."""
    base = {"root": tmp_path, "data": tmp_path / "data"}
    _seed_condition_outputs(base, ["old-a", "old-b", "old-c"])
    assert not condition_is_reusable(
        base, "natural", ("ce",), ("native",), ["new-a", "new-b", "new-c"]
    )


def test_condition_is_reusable_true_when_the_fingerprint_is_in_the_accepted_chain(
    tmp_path,
):
    base = {"root": tmp_path, "data": tmp_path / "data"}
    _seed_condition_outputs(base, ["old-a", "old-b", "old-c"])
    accepted = [{"new-a", "old-a"}, {"new-b", "old-b"}, {"new-c", "old-c"}]
    assert condition_is_reusable(
        base,
        "natural",
        ("ce",),
        ("native",),
        ["new-a", "new-b", "new-c"],
        accepted,
    )


def test_condition_is_reusable_false_for_a_legacy_artifact_with_no_recorded_fingerprint(
    tmp_path,
):
    base = {"root": tmp_path, "data": tmp_path / "data"}
    _seed_condition_outputs(base, [], with_fingerprint=False)
    assert not condition_is_reusable(base, "natural", ("ce",), ("native",), ["a", "b", "c"])
