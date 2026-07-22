from __future__ import annotations

import json
from typing import Any

from imbalance_benchmark.common import ensure_dirs, split_paths
from imbalance_benchmark.manifest.freeze import verify_manifest_freeze


def verify_resume_freezes(config: dict[str, Any]) -> None:
    """Verify every frozen split before resuming at tuning."""
    base = ensure_dirs(config)
    for index in range(3):
        path = split_paths(base, index)["data"] / "manifest_freeze.json"
        if not path.exists():
            raise FileNotFoundError(f"Cannot resume tuning without {path}")
        verify_manifest_freeze(json.loads(path.read_text()))
