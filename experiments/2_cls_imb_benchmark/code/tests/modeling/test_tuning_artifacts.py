from __future__ import annotations

from pathlib import Path

from imbalance_benchmark.modeling.workflows.tuning.candidate_registry import (
    load_registry,
    register_candidates,
    registry_lookup,
)
from imbalance_benchmark.modeling.workflows.tuning.tuning_artifacts import (
    ShardSpec,
    shard_path,
)


def test_shard_path_namespaces_by_round():
    root = Path("run")
    round0 = shard_path(root, ShardSpec("moderate", "ce", 1, "base", round=0))
    round1 = shard_path(root, ShardSpec("moderate", "ce", 1, "base", round=1))
    assert round0 != round1
    assert "round=0" in round0.parts
    assert "round=1" in round1.parts


def test_shard_path_defaults_to_round_zero():
    root = Path("run")
    default = shard_path(root, ShardSpec("moderate", "ce", 1, "base"))
    explicit = shard_path(root, ShardSpec("moderate", "ce", 1, "base", round=0))
    assert default == explicit


def test_registry_lookup_finds_a_previously_registered_candidate(tmp_path):
    configs = [{"lr": 1e-4}, {"lr": 3e-4}]
    register_candidates(tmp_path, "moderate", "ce", configs, round_index=0)
    registry = load_registry(tmp_path, "moderate")
    assert registry_lookup(registry, "ce", {"lr": 3e-4}) == (0, 1)


def test_registry_lookup_returns_none_for_an_unregistered_candidate(tmp_path):
    register_candidates(tmp_path, "moderate", "ce", [{"lr": 1e-4}], round_index=0)
    registry = load_registry(tmp_path, "moderate")
    assert registry_lookup(registry, "ce", {"lr": 3e-4}) is None


def test_register_candidates_is_idempotent_and_never_overwrites_the_original_round(
    tmp_path,
):
    register_candidates(tmp_path, "moderate", "ce", [{"lr": 1e-4}], round_index=0)
    register_candidates(tmp_path, "moderate", "ce", [{"lr": 1e-4}], round_index=1)
    registry = load_registry(tmp_path, "moderate")
    assert registry_lookup(registry, "ce", {"lr": 1e-4}) == (0, 0)


def test_register_candidates_honors_a_nonzero_start_index(tmp_path):
    register_candidates(
        tmp_path, "moderate", "ce", [{"lr": 3e-4}], round_index=1, start_index=3
    )
    registry = load_registry(tmp_path, "moderate")
    assert registry_lookup(registry, "ce", {"lr": 3e-4}) == (1, 3)


def test_registries_are_scoped_per_condition(tmp_path):
    register_candidates(tmp_path, "moderate", "ce", [{"lr": 1e-4}], round_index=0)
    severe_registry = load_registry(tmp_path, "severe")
    assert registry_lookup(severe_registry, "ce", {"lr": 1e-4}) is None
