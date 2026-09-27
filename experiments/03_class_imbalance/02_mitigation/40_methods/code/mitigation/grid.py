"""Flatten the config-declared arm x method x param grid for submit/fit dispatch.

``mitigation.grid`` in a run config lists the arms and, per method, the swept
parameter values (``null`` for a paramless method); this expands that into the
flat job lists ``submit`` and ``stage2`` (all pending) iterate over.
"""

from __future__ import annotations

from typing import Any, NamedTuple

__all__ = ["Stage1Job", "Stage2Job", "stage1_jobs", "stage2_jobs"]


class Stage1Job(NamedTuple):
    """One stage-one (arm, method, param) combination."""

    arm: str
    method: str
    param: float | None


class Stage2Job(NamedTuple):
    """One stage-two (arm, method, param) combination."""

    arm: str
    method: str
    param: float | None


def _params(values: list[Any]) -> list[float | None]:
    """Coerce a config's raw parameter list to floats, keeping ``null`` as ``None``."""
    return [None if value is None else float(value) for value in values]


def _jobs(config: dict[str, Any], key: str) -> list[tuple[str, str, float | None]]:
    grid = config.get("mitigation", {}).get("grid", {})
    arms = grid.get("arms", [])
    methods = grid.get(key, {})
    return [
        (arm, method, param)
        for arm in arms
        for method, values in methods.items()
        for param in _params(values)
    ]


def stage1_jobs(config: dict[str, Any]) -> list[Stage1Job]:
    """Every (arm, method, param) stage-one job the config's grid declares."""
    return [Stage1Job(*job) for job in _jobs(config, "stage1")]


def stage2_jobs(config: dict[str, Any]) -> list[Stage2Job]:
    """Every (arm, method, param) stage-two job the config's grid declares."""
    return [Stage2Job(*job) for job in _jobs(config, "stage2")]
