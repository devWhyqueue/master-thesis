from __future__ import annotations

from pathlib import Path

import pytest

from imbalance_benchmark.modeling.workflows.tuning.candidate_registry import (
    load_registry,
    load_round_grids,
    load_round_state,
    merge_round_state,
    register_candidates,
    registry_lookup,
    tuning_locked,
    write_round_grids,
)
from imbalance_benchmark.modeling.workflows.tuning.tuning_artifacts import (
    ShardSpec,
    shard_path,
    validate_shard_payload,
)


def _minimal_payload(fingerprint: list[str]) -> dict:
    return {
        "complete": True,
        "fingerprint": fingerprint,
        "seeds": [0],
        "scope_count": 1,
        "observation_keys": [{"scope_index": 0, "seed_index": 0, "seed": 0}],
        "cost_records": [{}],
    }


def test_validate_shard_payload_accepts_an_exact_fingerprint_match():
    validate_shard_payload(_minimal_payload(["a", "b", "c"]), ["a", "b", "c"])


def test_validate_shard_payload_rejects_an_unrelated_fingerprint():
    with pytest.raises(RuntimeError, match="fingerprint does not match"):
        validate_shard_payload(_minimal_payload(["old"] * 3), ["new"] * 3)


def test_validate_shard_payload_accepts_a_superseded_fingerprint_in_the_chain():
    """A shard tuned before a `method_grids` amendment must keep validating.

    ``accepted`` carries each split's superseded raw-file fingerprints
    (manifest_freeze.json's freed content_sha256 changes on amendment, so its
    whole-file hash changes too); an old shard's recorded fingerprint should
    still be recognized via that chain even though it no longer equals the
    freeze's current one.
    """
    payload = _minimal_payload(["old-0", "old-1", "old-2"])
    accepted = [{"new-0", "old-0"}, {"new-1", "old-1"}, {"new-2", "old-2"}]

    validate_shard_payload(payload, ["new-0", "new-1", "new-2"], accepted=accepted)


def test_validate_shard_payload_rejects_a_fingerprint_outside_the_chain():
    payload = _minimal_payload(["ancient-0", "ancient-1", "ancient-2"])
    accepted = [{"new-0", "old-0"}, {"new-1", "old-1"}, {"new-2", "old-2"}]

    with pytest.raises(RuntimeError, match="fingerprint does not match"):
        validate_shard_payload(payload, ["new-0", "new-1", "new-2"], accepted=accepted)


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


def test_round_grids_round_trip_signed(tmp_path):
    windows = {"ce": {"lr_window": [3e-4, 1e-3, 3e-3, 1e-2], "strength_window": None}}
    write_round_grids(tmp_path, "moderate", "base", 1, windows)
    loaded = load_round_grids(tmp_path, "moderate", "base")
    assert loaded == {"round": 1, "windows": windows}


def test_round_grids_are_scoped_per_phase(tmp_path):
    write_round_grids(tmp_path, "moderate", "base", 2, {"ce": {"lr_window": [1.0]}})
    with pytest.raises(RuntimeError, match="no signed lock"):
        load_round_grids(tmp_path, "moderate", "dependent")


def test_round_grids_reject_tampering(tmp_path):
    write_round_grids(tmp_path, "moderate", "base", 1, {"ce": {"lr_window": [1.0]}})
    path = tmp_path / "tuning_shards" / "tuning_round_grids_moderate_base.json"
    path.write_text(path.read_text().replace("1.0", "2.0"))
    with pytest.raises(RuntimeError, match="no longer matches"):
        load_round_grids(tmp_path, "moderate", "base")


def test_merge_round_state_combines_base_and_dependent_phases(tmp_path):
    merge_round_state(tmp_path, "moderate", {"ce": {"resolved": True, "tuning_limited": False}})
    merge_round_state(tmp_path, "moderate", {"crt": {"resolved": False, "tuning_limited": True}})
    state = load_round_state(tmp_path, "moderate")
    assert state["ce"]["resolved"] is True
    assert state["crt"]["tuning_limited"] is True


def test_merge_round_state_overwrites_a_methods_prior_entry(tmp_path):
    merge_round_state(tmp_path, "moderate", {"ce": {"resolved": False, "tuning_limited": False}})
    merge_round_state(tmp_path, "moderate", {"ce": {"resolved": True, "tuning_limited": False}})
    state = load_round_state(tmp_path, "moderate")
    assert state["ce"]["resolved"] is True


def test_tuning_locked_false_when_state_file_absent(tmp_path):
    assert tuning_locked(tmp_path, "moderate", ("ce",)) is False


def test_tuning_locked_false_when_a_required_method_is_missing(tmp_path):
    merge_round_state(tmp_path, "moderate", {"ce": {"resolved": True, "tuning_limited": False}})
    assert tuning_locked(tmp_path, "moderate", ("ce", "crt")) is False


def test_tuning_locked_false_while_a_method_is_still_shifting(tmp_path):
    merge_round_state(
        tmp_path, "moderate", {"ce": {"resolved": False, "tuning_limited": False}}
    )
    assert tuning_locked(tmp_path, "moderate", ("ce",)) is False


def test_tuning_locked_true_once_every_method_resolved_or_limited(tmp_path):
    merge_round_state(
        tmp_path,
        "moderate",
        {
            "ce": {"resolved": True, "tuning_limited": False},
            "crt": {"resolved": False, "tuning_limited": True},
        },
    )
    assert tuning_locked(tmp_path, "moderate", ("ce", "crt")) is True
