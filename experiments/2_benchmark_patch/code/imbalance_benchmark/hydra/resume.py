from __future__ import annotations

from dataclasses import dataclass
import json
from collections.abc import Callable
from typing import Any

from imbalance_benchmark.common import compute_sha256, ensure_dirs, split_paths
from imbalance_benchmark.manifest.freeze import verify_manifest_freeze
from imbalance_benchmark.modeling.context import CONTROLLED_CONDITIONS
from imbalance_benchmark.modeling.workflows.tuning.tuning_artifacts import (
    load_candidate,
)
from imbalance_benchmark.modeling.workflows.tuning.tuning_execution import (
    _bundle_indices,
)
from imbalance_benchmark.modeling.workflows.tuning.tuning_schedule import (
    bundled_array_size,
    bundled_observation_array_size,
    candidate_array_size,
    phase_methods,
    requested_shard,
)
from imbalance_benchmark.modeling.workflows.tuning.tuning_artifacts import (
    expected_observations,
)


@dataclass(frozen=True)
class ResumePlan:
    """Fingerprint-validated tuning work that still needs a SLURM allocation."""

    natural_indices: tuple[int, ...]
    controlled_indices: tuple[int, ...]


def verify_resume_freezes(config: dict[str, Any]) -> None:
    """Verify every frozen split before resuming at tuning."""
    base = ensure_dirs(config)
    for index in range(3):
        path = split_paths(base, index)["data"] / "manifest_freeze.json"
        if not path.exists():
            raise FileNotFoundError(f"Cannot resume tuning without {path}")
        verify_manifest_freeze(json.loads(path.read_text()))


def resume_plan(config: dict[str, Any]) -> ResumePlan:
    """Validate frozen artifacts and retain only incomplete base-stage array tasks."""
    verify_resume_freezes(config)
    base = ensure_dirs(config)
    freeze_paths = [
        split_paths(base, index)["data"] / "manifest_freeze.json" for index in range(3)
    ]
    freezes = [json.loads(path.read_text()) for path in freeze_paths]
    freeze = freezes[0]
    fingerprint = [compute_sha256(path) for path in freeze_paths]
    accepted = [
        {current, *split_freeze.get("superseded_freeze_file_hashes", [])}
        for current, split_freeze in zip(fingerprint, freezes)
    ]
    is_mil = config.get("dataset", {}).get("regime", "patch") == "wsi"
    natural_bundle, controlled_bundle = _bundle_sizes(config)
    return ResumePlan(
        _pending_natural(
            config, base, is_mil, freeze, fingerprint, natural_bundle, accepted
        ),
        _pending_controlled(
            base, is_mil, freeze, fingerprint, controlled_bundle, accepted
        ),
    )


def _bundle_sizes(config: dict[str, Any]) -> tuple[int, int]:
    slurm = config.get("slurm", {})
    controlled = int(slurm.get("tune_shards_per_task", 1))
    return int(slurm.get("tune_natural_shards_per_task", controlled)), controlled


def _pending_controlled(
    base: dict[str, Any],
    is_mil: bool,
    freeze: dict[str, Any],
    fingerprint: list[str],
    bundle_size: int,
    accepted: list[set[str]],
) -> tuple[int, ...]:
    assignments = tuple(freeze.get("tail_assignments", {"native": []}))
    total = len(CONTROLLED_CONDITIONS) * candidate_array_size(
        phase_methods(is_mil, "base", "balanced")
    )
    return tuple(
        index
        for index in range(bundled_array_size(total, bundle_size))
        if not _controlled_bundle_complete(
            index,
            bundle_size,
            total,
            is_mil,
            freeze,
            fingerprint,
            assignments,
            base,
            accepted,
        )
    )


def _pending_natural(
    config: dict[str, Any],
    base: dict[str, Any],
    is_mil: bool,
    freeze: dict[str, Any],
    fingerprint: list[str],
    bundle_size: int,
    accepted: list[set[str]],
) -> tuple[int, ...]:
    observations = int(
        config.get("slurm", {}).get("tune_natural_observations_per_candidate", 1)
    )
    natural_size = bundled_observation_array_size(
        candidate_array_size(phase_methods(is_mil, "base", "natural")),
        observations,
        bundle_size,
    )
    return tuple(
        index
        for index in range(natural_size)
        if not _natural_bundle_complete(
            index,
            bundle_size,
            observations,
            is_mil,
            freeze,
            fingerprint,
            base,
            accepted,
        )
    )


def _natural_bundle_complete(
    task_index: int,
    bundle_size: int,
    observations: int,
    is_mil: bool,
    freeze: dict[str, Any],
    fingerprint: list[str],
    base: dict[str, Any],
    accepted: list[set[str]],
) -> bool:
    def _load() -> None:
        for flat_index in _bundle_indices(task_index, bundle_size, observations, True):
            candidate_index, observation_index = divmod(flat_index, observations)
            spec = requested_shard(
                candidate_index,
                "base",
                "natural",
                is_mil,
                freeze["method_grids"],
                observation_index,
            )
            if spec is None:
                continue
            load_candidate(base["data"], spec, fingerprint, None, accepted)

    return _complete_or_missing(_load)


def _controlled_bundle_complete(
    task_index: int,
    bundle_size: int,
    total: int,
    is_mil: bool,
    freeze: dict[str, Any],
    fingerprint: list[str],
    assignments: tuple[str, ...],
    base: dict[str, Any],
    accepted: list[set[str]],
) -> bool:
    def _load() -> None:
        for shard_index in range(
            task_index * bundle_size, min((task_index + 1) * bundle_size, total)
        ):
            spec = requested_shard(
                shard_index, "base", "controlled", is_mil, freeze["method_grids"], None
            )
            if spec is None:
                continue
            load_candidate(
                base["data"],
                spec,
                fingerprint,
                expected_observations(spec.condition, assignments, freeze),
                accepted,
            )

    return _complete_or_missing(_load)


def _complete_or_missing(load: Callable[[], None]) -> bool:
    """Treat only absent artifacts as pending; corrupted artifacts stop resumption."""
    try:
        load()
    except RuntimeError as error:
        if str(error).startswith("Missing tuning"):
            return False
        raise
    return True
