from __future__ import annotations

import math
from typing import Any


def example_budget(support: int, reference_passes: int) -> int:
    """Return frozen example presentations for a training support."""
    return reference_passes * support


def examples_per_update(method: str, batch_size: int, cfg: dict[str, Any]) -> float:
    """Return examples consumed by one method-specific optimization step."""
    if method == "oko":
        return batch_size * (int(cfg["parameter"]) + 2)
    if method == "mde":
        return 2 * batch_size
    if method == "crt":
        return 1.2 * batch_size
    return float(batch_size)


def updates_for_exposure(
    example_budget: int, method: str, batch_size: int, cfg: dict[str, Any]
) -> int:
    """Derive update count needed to match frozen example presentations."""
    return math.ceil(example_budget / examples_per_update(method, batch_size, cfg))


def resolve_update_budget(
    ctx: dict[str, Any],
    method: str,
    cfg: dict[str, Any],
    batch_size: int,
    reference_passes: int,
) -> int:
    """Derive method/config-specific updates from frozen example presentations."""
    example_budget = int(
        ctx.get("example_budget", reference_passes * len(ctx["train_dataset"]))
    )
    return updates_for_exposure(example_budget, method, batch_size, cfg)
