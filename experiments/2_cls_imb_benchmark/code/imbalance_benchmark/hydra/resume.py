from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from imbalance_benchmark.common import compute_sha256, ensure_dirs, split_paths
from imbalance_benchmark.manifest.freeze import verify_manifest_freeze
from imbalance_benchmark.modeling.workflows.tuning.tuning_artifacts import (
    load_candidate,
)
from imbalance_benchmark.modeling.workflows.tuning.tuning_schedule import (
    bundled_array_size,
    candidate_array_size,
    phase_methods,
    requested_shard,
)
from imbalance_benchmark.modeling.workflows.tuning.tuning_artifacts import (
    expected_observations,
)
from imbalance_benchmark.modeling.workflows.tuning.tuning_reduction import reduce_phase


@dataclass(frozen=True)
class ResumePlan:
    """Fingerprint-validated tuning work that still needs a SLURM allocation."""

    base_natural_complete: bool
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
    freeze = json.loads(freeze_paths[0].read_text())
    fingerprint = [compute_sha256(path) for path in freeze_paths]
    is_mil = config.get("dataset", {}).get("regime", "patch") == "wsi"
    methods = phase_methods(is_mil, "base")
    assignments = tuple(freeze.get("tail_assignments", {"native": []}))
    natural_complete = _phase_complete(
        base, "natural", methods, freeze, fingerprint, assignments
    )
    bundle_size = int(config.get("slurm", {}).get("tune_shards_per_task", 1))
    total = 3 * candidate_array_size(methods)
    controlled = tuple(
        index
        for index in range(bundled_array_size(total, bundle_size))
        if not _controlled_bundle_complete(
            index, bundle_size, total, is_mil, freeze, fingerprint, assignments, base
        )
    )
    return ResumePlan(natural_complete, controlled)


def _phase_complete(
    base: dict[str, Any],
    condition: str,
    methods: tuple[str, ...],
    freeze: dict[str, Any],
    fingerprint: list[str],
    assignments: tuple[str, ...],
) -> bool:
    try:
        reduce_phase(
            base["data"],
            condition,
            "base",
            methods,
            freeze["method_grids"],
            fingerprint,
            expected_observations(condition, assignments, freeze),
        )
    except (OSError, ValueError, KeyError, RuntimeError):
        return False
    return True


def _controlled_bundle_complete(
    task_index: int,
    bundle_size: int,
    total: int,
    is_mil: bool,
    freeze: dict[str, Any],
    fingerprint: list[str],
    assignments: tuple[str, ...],
    base: dict[str, Any],
) -> bool:
    try:
        for shard_index in range(
            task_index * bundle_size, min((task_index + 1) * bundle_size, total)
        ):
            spec = requested_shard(
                shard_index, "base", "controlled", is_mil, freeze["method_grids"], None
            )
            assert spec is not None
            load_candidate(
                base["data"],
                spec,
                fingerprint,
                expected_observations("controlled", assignments, freeze),
            )
    except (OSError, ValueError, KeyError, RuntimeError):
        return False
    return True
