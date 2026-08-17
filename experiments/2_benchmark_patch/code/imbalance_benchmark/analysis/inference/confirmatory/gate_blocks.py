"""Loads one gate entry's paired prediction blocks for the crossed permutation tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from imbalance_benchmark.analysis.inference.confirmatory.holm import (
    MATCHED_CONTRAST_METHOD,
)
from imbalance_benchmark.analysis.metrics import assign_tiers
from imbalance_benchmark.analysis.query import load_seed_predictions, load_test_identity
from imbalance_benchmark.common import split_paths

__all__ = ["Arm", "Block", "load_freeze", "gate_blocks", "gate_tail_classes"]

Arm = np.ndarray | list[np.ndarray]
Block = tuple[np.ndarray, Arm, Arm, np.ndarray]


def load_freeze(paths: dict[str, Path]) -> dict[str, Any]:
    """Load the frozen analysis manifest, if `freeze` has already produced one."""
    freeze_path = paths["data"] / "manifest_freeze.json"
    return json.loads(freeze_path.read_text()) if freeze_path.exists() else {}


def _method_gate_blocks(
    entry: dict[str, Any], base_paths: dict[str, Path], is_mil: bool
) -> tuple[list[Block], dict[str, Any]] | None:
    """Load each split's paired method/CE prediction block for one gate entry."""
    blocks: list[Block] = []
    method_data: dict[str, Any] | None = None
    for index in range(3):
        paths = split_paths(base_paths, index)
        method = load_seed_predictions(
            paths, entry["severity"], entry["method"], entry["assignment"]
        )
        ce = load_seed_predictions(paths, entry["severity"], "ce", entry["assignment"])
        if method is None or ce is None:
            return None
        method_data = method
        id_df = load_test_identity(paths["data"] / "manifest.csv", is_mil)
        is_disc = entry["gate"] == "discrimination"
        blocks.append(
            (
                method["labels"],
                method["preds"] if is_disc else method["probs"],
                ce["preds"] if is_disc else ce["probs"],
                id_df["case_id"].to_numpy(),
            )
        )
    if method_data is None:
        return None
    return blocks, method_data


def _contrast_gate_blocks(
    entry: dict[str, Any], base_paths: dict[str, Path], is_mil: bool
) -> tuple[list[Block], dict[str, Any]] | None:
    """Load each split's matched-vs-unmatched prediction blocks for one gate entry."""
    matched, unmatched = (
        entry.get("matched_methods") or [],
        entry.get("unmatched_methods") or [],
    )
    if not matched or not unmatched:
        return None
    is_disc = entry["gate"] == "discrimination"
    blocks: list[Block] = []
    last_rec: dict[str, Any] | None = None
    for index in range(3):
        paths = split_paths(base_paths, index)
        arm_by_method: dict[str, Any] = {}
        labels: np.ndarray | None = None
        for method in (*matched, *unmatched):
            rec = load_seed_predictions(
                paths, entry["severity"], method, entry["assignment"]
            )
            if rec is None:
                return None
            arm_by_method[method] = rec["preds"] if is_disc else rec["probs"]
            labels, last_rec = rec["labels"], rec
        assert labels is not None
        matched_arm = [arm_by_method[method] for method in matched]
        unmatched_arm = [arm_by_method[method] for method in unmatched]
        id_df = load_test_identity(paths["data"] / "manifest.csv", is_mil)
        blocks.append((labels, matched_arm, unmatched_arm, id_df["case_id"].to_numpy()))
    if last_rec is None:
        return None
    return blocks, last_rec


def gate_blocks(
    entry: dict[str, Any], base_paths: dict[str, Path], is_mil: bool
) -> tuple[list[Block], dict[str, Any]] | None:
    """Load each split's paired prediction blocks for one gate entry."""
    if entry["method"] == MATCHED_CONTRAST_METHOD:
        return _contrast_gate_blocks(entry, base_paths, is_mil)
    return _method_gate_blocks(entry, base_paths, is_mil)


def gate_tail_classes(
    entry: dict[str, Any], base_paths: dict[str, Path], class_names: list[str]
) -> list[list[int]]:
    """Per-split tail-class indices for one gate entry's tail-NLL statistic."""
    tail_classes = []
    for index in range(3):
        fz = load_freeze(split_paths(base_paths, index))
        alloc = fz["assignment_conditions"][entry["assignment"]][entry["severity"]][
            "allocated_counts"
        ]
        tiers = assign_tiers(
            class_names,
            alloc,
            fz.get("tail_assignments", {}).get(entry["assignment"], class_names),
        )
        tail_classes.append(
            [idx for idx, name in enumerate(class_names) if tiers.get(name) == "tail"]
        )
    return tail_classes
