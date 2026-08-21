from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from imbalance_benchmark.common import sign_file, verify_signed_file, write_json
from imbalance_benchmark.modeling.workflows.tuning_aggregate import _selection_key
from imbalance_benchmark.modeling.workflows.tuning.tuning_artifacts import (
    ShardSpec,
    load_candidate,
    observation_key,
    write_atomic,
)


def registry_path(root: Path, condition: str) -> Path:
    """Path to one condition's cross-round candidate registry (not signed: a cache)."""
    return root / "tuning_shards" / f"candidate_registry_{condition}.json"


def _numeric(value: Any) -> float | None:
    """Normalize to float so ``1`` (oko's int grid) and ``1.0`` key identically."""
    return float(value) if value is not None else None


def _registry_key(method: str, config: dict[str, Any]) -> str:
    return f"{method}|{_numeric(config.get('parameter'))}|{_numeric(config.get('lr'))}"


def load_registry(root: Path, condition: str) -> dict[str, dict[str, int]]:
    """Load the map from (method, config) to the round/index it was first trained in.

    Legacy int-formatted keys (e.g. ``oko|1|...`` before parameter values
    were float-normalized) collapse onto today's canonical key; keep the
    lowest round, since that is what any prior decision resolved through.
    """
    path = registry_path(root, condition)
    registry = json.loads(path.read_text()) if path.exists() else {}
    normalized: dict[str, dict[str, int]] = {}
    for key, entry in registry.items():
        method, parameter, lr = key.split("|")
        config = {"lr": None if lr == "None" else float(lr)}
        if parameter != "None":
            config["parameter"] = float(parameter)
        canonical = _registry_key(method, config)
        existing = normalized.get(canonical)
        if existing is None or entry["round"] < existing["round"]:
            normalized[canonical] = entry
    return normalized


def registry_lookup(
    registry: dict[str, dict[str, int]], method: str, config: dict[str, Any]
) -> tuple[int, int] | None:
    """Find which round/index already trained this exact (method, config), if any."""
    entry = registry.get(_registry_key(method, config))
    return (entry["round"], entry["candidate_index"]) if entry else None


def registry_candidates_for_method(
    registry: dict[str, dict[str, int]], method: str
) -> list[dict[str, int]]:
    """Every round/index this method ever trained, regardless of its terminal window.

    Realized tuning cost must count each candidate once no matter how many
    rounds' windows it passed through, so this walks the whole registry
    instead of only the terminal active grid.
    """
    prefix = f"{method}|"
    return [entry for key, entry in registry.items() if key.startswith(prefix)]


def resolve_terminal_specs(
    root: Path,
    condition: str,
    phase: str,
    method: str,
    active_grid: list[dict[str, Any]],
) -> list[ShardSpec]:
    """Resolve one method's terminal active grid through the cross-round registry.

    Final reduction must never train or register a candidate itself: every
    value here has to already be registered from whichever round trained
    it, or the tuning lock was granted before that shard existed.
    """
    registry = load_registry(root, condition)
    specs = []
    for config in active_grid:
        found = registry_lookup(registry, method, config)
        if found is None:
            raise RuntimeError(f"Unregistered terminal candidate: {method} {config}")
        source_round, index = found
        specs.append(ShardSpec(condition, method, index, phase, round=source_round))
    return specs


def terminal_cost_payloads(
    root: Path,
    condition: str,
    phase: str,
    methods: tuple[str, ...],
    fingerprint: list[str],
    expected_observations: int | None = None,
    accepted: list[set[str]] | None = None,
) -> list[dict[str, Any]]:
    """Load every uniquely trained candidate across every round, for realized cost.

    A resolved method's terminal window only covers its last few candidates;
    realized cost must count every value trained on the way there, exactly
    once (the registry already dedupes a reused candidate).
    """
    registry = load_registry(root, condition)
    specs = []
    for method in methods:
        if method == "post_hoc_logit_adjustment":
            specs.append(ShardSpec(condition, method, 0, phase))
            continue
        specs.extend(
            ShardSpec(
                condition, method, entry["candidate_index"], phase, round=entry["round"]
            )
            for entry in registry_candidates_for_method(registry, method)
        )
    return [
        load_candidate(root, spec, fingerprint, expected_observations, accepted)
        for spec in specs
    ]


def register_candidates(
    root: Path,
    condition: str,
    method: str,
    configs: list[dict[str, Any]],
    round_index: int,
) -> None:
    """Record new candidates after this (method, round)'s already-claimed indices."""
    registry = load_registry(root, condition)
    prior = registry_candidates_for_method(registry, method)
    claimed = (e["candidate_index"] for e in prior if e["round"] == round_index)
    next_index = max(claimed, default=-1) + 1
    for config in configs:
        key = _registry_key(method, config)
        if key not in registry:
            registry[key] = {"round": round_index, "candidate_index": next_index}
            next_index += 1
    write_atomic(registry_path(root, condition), registry)


def _frozen_order_key(payload: dict[str, Any]) -> tuple[float, float]:
    """Order by (parameter, lr) so ties break the same regardless of trained round."""
    config = payload["config"]
    return (float(config.get("parameter", float("-inf"))), float(config["lr"]))


def select_candidate_payload(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    """Select a candidate in frozen order using the report's exact tie-break."""
    ordered = sorted(payloads, key=_frozen_order_key)
    selected, selected_key = ordered[0], None
    for payload in ordered:
        observations = sorted(payload["metrics"], key=observation_key)
        metrics = [
            (
                float(metric["balanced_accuracy"]),
                float(metric["macro_f1"]),
                float(metric["nll"]),
            )
            for metric in observations
        ]
        key = _selection_key(metrics)
        if selected_key is None or key > selected_key:
            selected, selected_key = payload, key
    return selected


def round_grids_path(root: Path, condition: str, phase: str) -> Path:
    """Path to one condition/phase's signed current-round active-window record.

    Base and dependent methods can be on different round numbers at once
    (dependent only starts once CE resolves), so each phase tracks its own
    round independently.
    """
    return root / "tuning_shards" / f"tuning_round_grids_{condition}_{phase}.json"


def write_round_grids(
    root: Path,
    condition: str,
    phase: str,
    round_index: int,
    windows: dict[str, dict[str, Any]],
) -> Path:
    """Sign this round's active lr/strength windows so shard, reduce, and decide agree.

    ``windows`` maps each method still under search to
    ``{"lr_window": [...], "strength_window": [...] | None}``; a method
    already resolved or tuning-limited is simply absent.
    """
    path = round_grids_path(root, condition, phase)
    write_json(path, {"round": round_index, "windows": windows})
    sign_file(path)
    return path


def load_round_grids(root: Path, condition: str, phase: str) -> dict[str, Any]:
    """Load and verify the current round's signed active windows."""
    path = round_grids_path(root, condition, phase)
    verify_signed_file(path)
    return json.loads(path.read_text())


def round_state_path(root: Path, condition: str) -> Path:
    """Path to one condition's signed tuning lock, covering every phase's methods."""
    return root / "tuning_shards" / f"tuning_round_state_{condition}.json"


def merge_round_state(root: Path, condition: str, updates: dict[str, Any]) -> Path:
    """Sign an updated tuning lock, merging one phase's methods into any prior state."""
    path = round_state_path(root, condition)
    state = load_round_state(root, condition) if path.exists() else {}
    state.update(updates)
    write_json(path, state)
    sign_file(path)
    return path


def load_round_state(root: Path, condition: str) -> dict[str, Any]:
    """Load and verify the current signed tuning lock."""
    path = round_state_path(root, condition)
    verify_signed_file(path)
    return json.loads(path.read_text())


def tuning_locked(root: Path, condition: str, methods: tuple[str, ...]) -> bool:
    """True once every required method is resolved or correctly marked tuning-limited.

    This is the tuning lock confirmation must check: a method's mere
    presence in a selection file is not enough, since a round's winner
    can still be an unresolved edge case awaiting another round.
    """
    path = round_state_path(root, condition)
    if not path.exists():
        return False
    state = load_round_state(root, condition)
    return all(
        method in state
        and (state[method]["resolved"] or state[method]["tuning_limited"])
        for method in methods
    )
