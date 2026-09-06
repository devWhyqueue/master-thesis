"""Shared constants and per-split plumbing for the analyze/ subpackage."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from imbalance_benchmark.analysis.inference.context import (
    BootstrapContext,
    _tail_classes,
)
from imbalance_benchmark.common import ensure_dirs, split_paths

__all__ = [
    "CE",
    "UNMATCHED_METHOD",
    "MATCHED_METHOD",
    "N_REPLICATES",
    "N_PERMUTATIONS",
    "ENDPOINTS",
    "read_json",
    "fixed_tail_classes",
    "endpoint_distribution",
    "iter_splits",
]

CE = "ce"
UNMATCHED_METHOD = "weighted_ce"  # reads prevalence
MATCHED_METHOD = "semantic_scale_ce"  # reads diversity
N_REPLICATES = 10_000
N_PERMUTATIONS = 100_000
ENDPOINTS = ("ba", "tail_nll")


def read_json(path: Path) -> dict[str, Any]:
    """Read one JSON file as a plain dict."""
    return json.loads(path.read_text(encoding="utf-8"))


def fixed_tail_classes(freeze: dict[str, Any], class_names: list[str]) -> list[int]:
    """Tail group fixed once from the severe allocation, reused for balanced (Sec. Endpoints).

    Allocated counts are, by the build-stage invariants, identical across
    the three diversity levels, so any level string gives the same tail set;
    'random' is used here only because it is guaranteed present.
    """
    return _tail_classes(freeze, class_names, "random", "severe")


def endpoint_distribution(
    ctx: BootstrapContext,
    endpoint: str,
    preds: dict[str, Any],
    n_classes: int,
    tail_classes: list[int],
) -> np.ndarray | None:
    """One arm's bootstrap distribution for the requested endpoint."""
    if endpoint == "ba":
        return ctx.ba_distribution(preds["labels"], preds["preds"], n_classes)
    return ctx.tail_nll_distribution(preds["labels"], preds["probs"], tail_classes)


def iter_splits(
    config: dict[str, Any],
) -> Iterator[tuple[int, dict[str, Path], dict[str, Any]]]:
    """Yield ``(split_index, exp3_paths, freeze)`` for every split with a derived freeze."""
    for split_index in range(3):
        exp3_paths = split_paths(ensure_dirs(config), split_index)
        freeze_path = exp3_paths["data"] / "manifest_freeze.json"
        if freeze_path.exists():
            yield split_index, exp3_paths, read_json(freeze_path)
