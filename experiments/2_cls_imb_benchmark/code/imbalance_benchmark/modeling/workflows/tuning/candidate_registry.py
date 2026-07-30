from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from imbalance_benchmark.common import sign_file, verify_signed_file, write_json
from imbalance_benchmark.modeling.workflows.tuning_aggregate import _selection_key
from imbalance_benchmark.modeling.workflows.tuning.tuning_artifacts import (
    observation_key,
    write_atomic,
)

__all__ = [
    "registry_path",
    "load_registry",
    "registry_lookup",
    "register_candidates",
    "select_candidate_payload",
    "round_grids_path",
    "write_round_grids",
    "load_round_grids",
    "round_state_path",
    "merge_round_state",
    "load_round_state",
    "tuning_locked",
]


def registry_path(root: Path, condition: str) -> Path:
    """Path to one condition's cross-round candidate registry (not signed: a cache)."""
    return root / "tuning_shards" / f"candidate_registry_{condition}.json"


def _registry_key(method: str, config: dict[str, Any]) -> str:
    return f"{method}|{config.get('parameter')}|{config.get('lr')}"


def load_registry(root: Path, condition: str) -> dict[str, dict[str, int]]:
    """Load the map from (method, config) to the round/index it was first trained in."""
    path = registry_path(root, condition)
    return json.loads(path.read_text()) if path.exists() else {}


def registry_lookup(
    registry: dict[str, dict[str, int]], method: str, config: dict[str, Any]
) -> tuple[int, int] | None:
    """Find which round/index already trained this exact (method, config), if any."""
    entry = registry.get(_registry_key(method, config))
    return (entry["round"], entry["candidate_index"]) if entry else None


def register_candidates(
    root: Path,
    condition: str,
    method: str,
    configs: list[dict[str, Any]],
    round_index: int,
    start_index: int = 0,
) -> None:
    """Record where one round's freshly trained candidates now live, idempotently."""
    registry = load_registry(root, condition)
    for offset, config in enumerate(configs):
        key = _registry_key(method, config)
        registry.setdefault(
            key, {"round": round_index, "candidate_index": start_index + offset}
        )
    write_atomic(registry_path(root, condition), registry)


def _frozen_order_key(payload: dict[str, Any]) -> tuple[float, float]:
    """Order by (parameter, lr): value-based so ties break the same way regardless
    of which adaptive-search round a candidate was actually trained in."""
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
