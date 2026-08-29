from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PilotConstraints:
    """Allocation and independent-unit floors frozen from the pilot."""

    patch_floor: int
    independent_floor: int


def _pilot_constraints(pilot_report_path: Path) -> PilotConstraints:
    """Freeze the pilot's independent-unit floor as both the unit and count floor.

    The per-patient quota is the scarcest class's *minimum* per-patient
    inventory at the largest pilot level, so one patient holding a single
    patch could move the frozen floor - and every condition's achievable
    severity - by an order of magnitude, making splits incomparable. Per-class
    independent support is instead guaranteed directly by ``independent_floor``
    in pool designation plus the contribution caps.
    """
    if not pilot_report_path.exists():
        return PilotConstraints(10, 10)
    definitive_floor = json.loads(pilot_report_path.read_text())["definitive_floor"]
    return PilotConstraints(definitive_floor, definitive_floor)
