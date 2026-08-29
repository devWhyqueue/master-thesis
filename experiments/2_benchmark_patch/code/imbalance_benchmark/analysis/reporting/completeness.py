from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from imbalance_benchmark.common import split_paths
from imbalance_benchmark.modeling.context import roster_for_regime


def _split_assignment_severities(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    conditions = json.loads(path.read_text()).get("assignment_conditions", {})
    return {
        (assignment, severity)
        for assignment, severities in conditions.items()
        for severity in severities
    }


def expected_comparison_keys(
    base_paths: dict[str, Path], config: dict[str, Any] | None
) -> set[tuple[str, str, str, str]]:
    """Return the frozen method-by-gate roster expected from every split.

    ``(assignment, severity)`` pairs are read from each split's actual
    ``assignment_conditions``, not a fixed severity tuple: which assignments
    carry a narrowed condition is per-dataset (plans/04-crossed-condition-
    family.md), so a hardcoded ("moderate", "severe") would either miss the
    narrowed pair or wrongly expect it everywhere.
    """
    is_mil = config.get("dataset", {}).get("regime") == "wsi" if config else False
    methods = roster_for_regime(is_mil)
    assignment_severities: set[tuple[str, str]] = set()
    for index in range(3):
        path = split_paths(base_paths, index)["data"] / "manifest_freeze.json"
        assignment_severities |= _split_assignment_severities(path)
    return {
        (assignment, severity, method, gate)
        for assignment, severity in assignment_severities
        for method in methods
        for gate in ("discrimination", "calibration")
    }
