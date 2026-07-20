from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch.utils.data

from imbalance_benchmark.modeling.context import Regime

__all__ = ["RunContext"]


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
