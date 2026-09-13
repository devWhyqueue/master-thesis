from __future__ import annotations

import json
from pathlib import Path

from imbalance_benchmark.common import split_paths


def _achieved_rho_by_split(
    base_paths: dict[str, Path],
) -> dict[tuple[str, str], dict[int, float]]:
    """Achieved ρ per (assignment, severity), keyed by split index.

    Empty if any split has not been frozen yet - nothing to compare.
    """
    freeze_paths = [
        split_paths(base_paths, index)["data"] / "manifest_freeze.json"
        for index in range(3)
    ]
    if not all(path.exists() for path in freeze_paths):
        return {}
    achieved: dict[tuple[str, str], dict[int, float]] = {}
    for index, freeze_path in enumerate(freeze_paths):
        freeze = json.loads(freeze_path.read_text())
        for assignment, conditions in freeze["assignment_conditions"].items():
            for severity, condition in conditions.items():
                achieved.setdefault((assignment, severity), {})[index] = condition[
                    "achieved_rho"
                ]
    return achieved


def require_consistent_achieved_severity(
    base_paths: dict[str, Path], relative_tolerance: float = 0.5
) -> None:
    """Refuse to average splits whose achieved ρ differ materially.

    A split whose severity construction collapsed toward balanced must not
    be averaged in with splits that achieved the real requested ratio: the
    three-split table would then report an effect that is part real, part
    null, attributed to neither. Freezing already rejects a degenerate split
    on its own (see ``reject_degenerate_conditions``); this is defense in
    depth for manifests frozen before that guard existed.
    """
    for (assignment, severity), per_split in _achieved_rho_by_split(base_paths).items():
        values = list(per_split.values())
        spread = (max(values) - min(values)) / max(values)
        if spread > relative_tolerance:
            raise RuntimeError(
                f"Achieved ρ for {assignment}/{severity} differs materially across "
                f"splits and cannot be averaged: {per_split}"
            )
