from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from imbalance_benchmark.analysis.inference.confirmatory.holm import PRIMARY_METHODS
from imbalance_benchmark.analysis.predictors.rq3_analysis import _rq3_group
from imbalance_benchmark.common import (
    ensure_dirs,
    load_config,
    output_root,
    sign_file,
    verify_signed_file,
    write_json,
)
from imbalance_benchmark.manifest.freeze import accepted_freeze_hashes

__all__ = ["build_matching_record", "cmd_match", "unit_key"]

logger = logging.getLogger(__name__)

# Prespecified matched members per dominant shortage (protocol app:testing).
MATCHED_MEMBERS: dict[str, frozenset[str]] = {
    "nominal": frozenset({"weighted_ce", "class_balanced_ce"}),
    "independent": frozenset({"independent_support_ce"}),
    "difficulty": frozenset({"pilot_difficulty_ce"}),
    "diversity": frozenset({"semantic_scale_ce"}),
}
AMBIGUITY_BAND_SD = 0.25


def unit_key(group: str, assignment: str, severity: str) -> str:
    """Stable string key for one dataset-target x severity x tail-assignment unit."""
    return f"{group}::{assignment}::{severity}"


def _load_root_units(root: Path) -> list[dict[str, Any]]:
    """Average one dataset-regime root's per-split shortage scores across its 3 splits."""
    per_unit: dict[tuple[str, str], list[dict[str, float]]] = {}
    group: str | None = None
    freeze_hash: str | None = None
    for index in range(3):
        split_dir = root / f"split={index}" / "data"
        freeze_path, profile_path = (
            split_dir / "manifest_freeze.json",
            split_dir / "signal_profile.json",
        )
        if not freeze_path.exists() or not profile_path.exists():
            raise RuntimeError(
                f"Matching requires signals for all three splits; missing under {split_dir}"
            )
        verify_signed_file(profile_path)
        freeze = json.loads(freeze_path.read_text())
        profile = json.loads(profile_path.read_text())
        if profile.get("freeze_content_sha256") not in accepted_freeze_hashes(freeze):
            raise RuntimeError(
                f"{profile_path} is stale relative to its freeze; re-run signals"
            )
        group, freeze_hash = _rq3_group(freeze), freeze.get("content_sha256")
        for comparison in profile["comparisons"]:
            key = (comparison["assignment"], comparison["severity"])
            per_unit.setdefault(key, []).append(comparison)
    if group is None:
        return []
    return [
        {
            "group": group,
            "assignment": assignment,
            "severity": severity,
            "freeze_content_sha256": freeze_hash,
            "nominal_shortage": float(np.mean([r["nominal_shortage"] for r in rows])),
            "independent_shortage": float(
                np.mean([r["independent_shortage"] for r in rows])
            ),
            "diversity_shortage": float(
                np.mean([r["diversity_shortage"] for r in rows])
            ),
            "support_difficulty_alignment": float(
                np.mean([r["support_difficulty_alignment"] for r in rows])
            ),
        }
        for (assignment, severity), rows in per_unit.items()
    ]


def _standardize(values: np.ndarray) -> np.ndarray:
    std = values.std(ddof=0)
    return (values - values.mean()) / std if std > 0 else np.zeros_like(values)


def _standardize_scores(units: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    """Standardize each of the four shortage scores across every pooled unit."""
    raw = {
        "nominal": np.array([unit["nominal_shortage"] for unit in units]),
        "independent": np.array([unit["independent_shortage"] for unit in units]),
        "difficulty": np.array(
            [-unit["support_difficulty_alignment"] for unit in units]
        ),
        "diversity": np.array([unit["diversity_shortage"] for unit in units]),
    }
    return {label: _standardize(values) for label, values in raw.items()}


def _label_unit(unit: dict[str, Any], scores: dict[str, float]) -> dict[str, Any]:
    """Dominant/ambiguous label and matched/unmatched method sets for one unit."""
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    ambiguous = (ranked[0][1] - ranked[1][1]) < AMBIGUITY_BAND_SD
    dominant = None if ambiguous else ranked[0][0]
    matched = MATCHED_MEMBERS[dominant] if dominant else frozenset()
    unmatched = PRIMARY_METHODS - matched
    return {
        "group": unit["group"],
        "assignment": unit["assignment"],
        "severity": unit["severity"],
        "freeze_content_sha256": unit["freeze_content_sha256"],
        "standardized_scores": scores,
        "dominant": dominant,
        "ambiguous": ambiguous,
        "matched_methods": sorted(matched),
        "unmatched_methods": sorted(unmatched),
    }


def build_matching_record(roots: list[Path]) -> dict[str, Any]:
    """Standardize the four shortage scores across every pooled unit and label each.

    Protocol app:testing: the dominant shortage is the largest standardized
    score; a unit whose two largest scores lie within 0.25 SD is ambiguous
    and carries no matched label.
    """
    units = [unit for root in roots for unit in _load_root_units(root)]
    if not units:
        return {"units": {}, "roots": [str(root) for root in roots]}
    standardized = _standardize_scores(units)
    records = {
        unit_key(unit["group"], unit["assignment"], unit["severity"]): _label_unit(
            unit,
            {label: float(column[index]) for label, column in standardized.items()},
        )
        for index, unit in enumerate(units)
    }
    return {"units": records, "roots": [str(root) for root in roots]}


def cmd_match(args: argparse.Namespace) -> None:
    """Build the prespecified cross-dataset matching record over every listed root."""
    config = load_config(args.config)
    roots = [Path(r) for r in config.get("rq3", {}).get("dataset_roots", [])] or [
        output_root(config)
    ]
    base_paths = ensure_dirs(config)
    record = build_matching_record(roots)
    path = base_paths["data"] / "matching_record.json"
    write_json(path, record)
    sign_file(path)
    logger.info("match: wrote %s (%d units)", path, len(record["units"]))
