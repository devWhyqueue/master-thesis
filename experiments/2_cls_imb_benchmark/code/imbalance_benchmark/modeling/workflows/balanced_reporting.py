"""Assignment-specific reporting copies for one shared balanced prediction set."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from imbalance_benchmark.analysis.metrics import assign_tiers
from imbalance_benchmark.analysis.predictors.tier_summaries import tier_metrics
from imbalance_benchmark.common import read_run_record, write_run_record

_TIER_ARRAYS = ("precision", "recall", "f1", "support", "nll", "brier")


def copy_balanced_tier_summaries(
    paths: dict[str, Path], freeze: dict[str, Any]
) -> None:
    """Copy balanced records for each assignment, recalculating only tier summaries."""
    source_root = paths["results"] / "assignment=native" / "balanced"
    for source in source_root.rglob("run.json"):
        record = read_run_record(source.parent)
        if record is None:
            continue
        for assignment in freeze.get("tail_assignments", {}):
            if assignment == "native":
                continue
            copied = _retier_record(record, freeze, assignment)
            destination = _assignment_path(paths["results"], source, assignment)
            write_run_record(destination, copied)


def _retier_record(
    record: dict[str, Any], freeze: dict[str, Any], assignment: str
) -> dict[str, Any]:
    """Recalculate balanced head/body/tail summaries without altering predictions."""
    copied = deepcopy(record)
    class_names = list(copied.get("class_names", []))
    tiers = _balanced_tiers(freeze, assignment, class_names)
    copied["assignment"] = assignment
    for payload in copied.get("splits", {}).values():
        if isinstance(payload, dict):
            payload["tier_metrics"] = _tier_metrics(payload, class_names, tiers)
    return copied


def _balanced_tiers(
    freeze: dict[str, Any], assignment: str, class_names: list[str]
) -> dict[str, str]:
    """Resolve balanced ties with the semantic order of one locked assignment."""
    counts = freeze["conditions"]["balanced"]["allocated_counts"]
    order = freeze["tail_assignments"][assignment]
    return assign_tiers(class_names, counts, order)


def _tier_metrics(
    payload: dict[str, Any], class_names: list[str], tiers: dict[str, str]
) -> dict[str, dict[str, Any]]:
    """Summarize stored classwise values under one assignment's tiers."""
    stats = {
        name: np.asarray(payload.get(f"{name}_per_class", []), dtype=float)
        for name in _TIER_ARRAYS
    }
    return tier_metrics(class_names, tiers, stats)


def _assignment_path(results: Path, source: Path, assignment: str) -> Path:
    """Map one native balanced record path to its assignment-specific copy."""
    method = source.parent.parent.name
    seed = source.parent.name
    return results / f"assignment={assignment}" / "balanced" / method / seed
