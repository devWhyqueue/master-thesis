from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from imbalance_benchmark.common import split_paths
from imbalance_benchmark.modeling.context import roster_for_regime


def expected_comparison_keys(
    base_paths: dict[str, Path], config: dict[str, Any] | None
) -> set[tuple[str, str, str, str]]:
    """Return the frozen method-by-gate roster expected from every split."""
    is_mil = config.get("dataset", {}).get("regime") == "wsi" if config else False
    methods = roster_for_regime(is_mil)
    assignments = set()
    for index in range(3):
        path = split_paths(base_paths, index)["data"] / "manifest_freeze.json"
        if path.exists():
            assignments.update(json.loads(path.read_text()).get("tail_assignments", {}))
    return {
        (assignment, severity, method, gate)
        for assignment in assignments
        for severity in ("moderate", "severe")
        for method in methods
        for gate in ("discrimination", "calibration")
    }
