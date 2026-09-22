from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any

import torch
import torch.utils.data

from imbalance_benchmark.modeling.context import Regime

__all__ = ["RunContext", "param_counts", "RunExposure", "cost_payload", "updates_for"]


@dataclass
class RunContext(Regime):
    """Shared per-condition confirmation inputs: locked val/test loaders, paths, and seeds."""

    val_loader: torch.utils.data.DataLoader
    test_loader: torch.utils.data.DataLoader
    paths: dict[str, Path]
    seeds: list[int]
    class_names: list[str]
    assignment: str
    feature_provenance: dict[str, str] | None = field(default=None, kw_only=True)


def param_counts(model: torch.nn.Module) -> dict[str, int]:
    """Total and trainable parameter counts for one confirmation run's cost record."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_parameters": int(total), "trainable_parameters": int(trainable)}


def _peak_memory(override: int | None) -> int:
    """Return a measured peak unless a non-training method supplies its exact zero."""
    if override is not None:
        return int(override)
    return int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0


@dataclass(frozen=True)
class RunExposure:
    """Exactly what one fitted run consumed, as recorded by its training loop."""

    unique_examples: int
    exposed_examples: int
    processed_examples: int
    training_footprint_parameters: int | None = None
    peak_memory_bytes: int | None = None
    processed_instances: int | None = None


def cost_payload(
    method: str,
    budget: int,
    elapsed: float,
    model: torch.nn.Module,
    exposure: RunExposure,
) -> dict[str, Any]:
    """Build an exact confirmation cost record from the examples actually consumed."""
    updates = updates_for(method, budget)
    peak = _peak_memory(exposure.peak_memory_bytes)
    counts = param_counts(model)
    if method == "post_hoc_logit_adjustment":
        counts["trainable_parameters"] = 0
    if exposure.training_footprint_parameters is not None:
        counts["training_footprint_parameters"] = exposure.training_footprint_parameters
    processed = exposure.processed_examples
    payload = {
        "updates": updates,
        "processed_examples": processed,
        "wall_clock_seconds": elapsed,
        "accelerator_hours": elapsed / 3600 if torch.cuda.is_available() else 0.0,
        "peak_accelerator_memory_bytes": peak,
        "examples_per_update": processed / updates if updates else 0.0,
        "unique_training_examples": exposure.unique_examples,
        "unique_examples_exposed": exposure.exposed_examples,
        "effective_passes_through_unique_examples": processed
        / max(exposure.unique_examples, 1),
        **counts,
    }
    if exposure.processed_instances is not None:
        payload["processed_instances"] = exposure.processed_instances
    return payload


def updates_for(method: str, budget: int) -> int:
    """Report's update accounting: cRT adds its stage-two budget."""
    if method == "crt":
        return budget + math.ceil(0.2 * budget)
    if method == "post_hoc_logit_adjustment":
        return 0
    return budget
