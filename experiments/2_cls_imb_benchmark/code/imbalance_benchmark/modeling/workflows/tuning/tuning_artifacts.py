from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import os
from pathlib import Path
from typing import Any

from imbalance_benchmark.common import split_paths, verify_signed_file


@dataclass(frozen=True)
class ShardSpec:
    """One independently executable frozen candidate or observation.

    ``round`` namespaces an adaptive search round (0 is the frozen initial
    window); a candidate keeps the same ``round``/``candidate_index`` for as
    long as it was actually trained in, so a later round that reuses it
    reads the same file instead of retraining it (see the candidate
    registry in this module).
    """

    condition: str
    method: str
    candidate_index: int
    phase: str
    observation_index: int | None = None
    round: int = 0


def validate_shard_payload(
    payload: dict[str, Any],
    fingerprint: list[str],
    spec: ShardSpec | None = None,
) -> None:
    """Reject incomplete, stale, or misaddressed shard output."""
    if not payload.get("complete"):
        raise RuntimeError("Tuning shard is incomplete")
    if payload.get("fingerprint") != fingerprint:
        raise RuntimeError("Tuning shard freeze fingerprint does not match")
    if spec is not None and payload.get("spec") != asdict(spec):
        raise RuntimeError("Tuning shard specification does not match its path")
    _validate_observations(payload)


def _validate_observations(payload: dict[str, Any]) -> None:
    seeds = payload.get("seeds")
    scope_count = payload.get("scope_count")
    observations = payload.get("observation_keys")
    if not isinstance(seeds, list) or not isinstance(scope_count, int):
        raise RuntimeError("Tuning shard observation metadata is incomplete")
    complete_set = [
        {"scope_index": scope_index, "seed_index": seed_index, "seed": seed}
        for scope_index in range(scope_count)
        for seed_index, seed in enumerate(seeds)
    ]
    index = payload.get("spec", {}).get("observation_index")
    if index is not None and (
        not isinstance(index, int) or index < 0 or index >= len(complete_set)
    ):
        raise RuntimeError("Tuning shard observation index is out of range")
    expected = complete_set if index is None else complete_set[index : index + 1]
    identities = [_identity(item) for item in observations or []]
    if identities != expected or len(
        {tuple(item.values()) for item in identities}
    ) != len(identities):
        raise RuntimeError("Tuning shard observations are missing or duplicated")
    metrics = payload.get("metrics")
    if metrics is not None:
        metric_ids = sorted((_identity(item) for item in metrics), key=observation_key)
        if metric_ids != expected:
            raise RuntimeError("Tuning shard metrics do not match its observations")
    if len(payload.get("cost_records", [])) != len(expected):
        raise RuntimeError("Tuning shard cost records do not match its observations")


def _identity(observation: dict[str, Any]) -> dict[str, Any]:
    return {key: observation[key] for key in ("scope_index", "seed_index", "seed")}


def observation_key(observation: dict[str, Any]) -> tuple[int, int]:
    """Return the canonical split/assignment scope then seed ordering key."""
    return int(observation["scope_index"]), int(observation["seed_index"])


def shard_path(root: Path, spec: ShardSpec) -> Path:
    """Return the collision-free artifact path for one candidate or observation."""
    candidate = (
        root
        / "tuning_shards"
        / spec.condition
        / spec.phase
        / spec.method
        / f"round={spec.round}"
    )
    if spec.observation_index is None:
        return candidate / f"candidate={spec.candidate_index}.json"
    return (
        candidate
        / f"candidate={spec.candidate_index}"
        / f"observation={spec.observation_index}.json"
    )


def write_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Replace one shard artifact only after its JSON is complete."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def load_candidate(
    root: Path,
    spec: ShardSpec,
    fingerprint: list[str],
    expected_observations: int | None,
) -> dict[str, Any]:
    """Load one candidate artifact or its exact observation-shard equivalent."""
    path = shard_path(root, spec)
    observation_dir = path.with_suffix("")
    if path.exists():
        if observation_dir.exists():
            raise RuntimeError(f"Duplicate candidate and observation shards: {path}")
        payload = json.loads(path.read_text())
        validate_shard_payload(payload, fingerprint, spec)
        return payload
    if expected_observations is None:
        raise RuntimeError(f"Missing tuning shard: {path}")
    return _merge_observation_shards(root, spec, fingerprint, expected_observations)


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


def condition_is_reusable(
    base: dict[str, Path],
    condition: str,
    methods: tuple[str, ...],
    assignments: tuple[str, ...],
) -> bool:
    """Accept a completed serial condition only when every split is identical."""
    selections, costs = [], []
    for index in range(3):
        data = split_paths(base, index)["data"]
        selection = data / f"tuning_selections_{condition}.json"
        cost = data / f"tuning_search_cost_{condition}.json"
        try:
            verify_signed_file(selection)
            selections.append(json.loads(selection.read_text()))
            costs.append(json.loads(cost.read_text()))
        except (FileNotFoundError, json.JSONDecodeError, RuntimeError):
            return False
    if selections[1:] != selections[:1] * 2 or costs[1:] != costs[:1] * 2:
        return False
    required_cost = {"wall_clock_seconds", "accelerator_hours", "processed_examples"}
    if not required_cost.issubset(costs[0]):
        return False
    scoped = ("native",) if condition in {"natural", "balanced"} else assignments
    return all(
        all(
            isinstance(selections[0].get(name, {}).get(condition, {}).get(method), dict)
            for method in methods
        )
        for name in scoped
    )


def expected_observations(
    condition: str, assignments: tuple[str, ...], freeze: dict[str, Any]
) -> int:
    """Count the frozen assignment, split, and initialization observations."""
    assignment_count = 1 if condition in {"natural", "balanced"} else len(assignments)
    seed_count = sum(
        key.startswith("tuning_initialization_") for key in freeze["seed_roles"]
    )
    return assignment_count * 3 * seed_count


def selected_ce(root: Path, condition: str) -> dict[str, Any]:
    """Load the signed base-phase CE selection for dependent shards."""
    path = root / "tuning_shards" / f"base_selections_{condition}.json"
    verify_signed_file(path)
    return json.loads(path.read_text())["ce"]


def _merge_observation_shards(
    root: Path, spec: ShardSpec, fingerprint: list[str], expected: int
) -> dict[str, Any]:
    observed_specs = [
        replace(spec, observation_index=index) for index in range(expected)
    ]
    paths = [shard_path(root, observed) for observed in observed_specs]
    directory = paths[0].parent
    if not directory.exists() or set(directory.glob("*.json")) != set(paths):
        raise RuntimeError(f"Missing or duplicate tuning observations: {directory}")
    payloads = []
    for observed, path in zip(observed_specs, paths, strict=True):
        payload = json.loads(path.read_text())
        validate_shard_payload(payload, fingerprint, observed)
        payloads.append(payload)
    return _merged_payload(spec, payloads)


def _merged_payload(spec: ShardSpec, payloads: list[dict[str, Any]]) -> dict[str, Any]:
    if any(payload["config"] != payloads[0]["config"] for payload in payloads[1:]):
        raise RuntimeError("Observation shards disagree on the frozen candidate")
    return {
        "candidate_index": spec.candidate_index,
        "config": payloads[0]["config"],
        "metrics": [metric for payload in payloads for metric in payload["metrics"]],
        "cost_records": [
            record for payload in payloads for record in payload["cost_records"]
        ],
        "started_at": min(payload["started_at"] for payload in payloads),
        "completed_at": max(payload["completed_at"] for payload in payloads),
        "accelerator_seconds": sum(
            payload["accelerator_seconds"] for payload in payloads
        ),
        "peak_accelerator_memory_bytes": max(
            payload["peak_accelerator_memory_bytes"] for payload in payloads
        ),
        "hardware": {"observation_shards": [p["hardware"] for p in payloads]},
    }
