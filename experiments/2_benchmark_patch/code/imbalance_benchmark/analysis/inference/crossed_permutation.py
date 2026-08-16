"""Permutation tests for the three fixed patient-split repetitions."""

from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Callable
from typing import Any, cast

import numpy as np

from imbalance_benchmark.analysis.inference.holm import PRIMARY_METHODS
from imbalance_benchmark.analysis.inference.permutation import (
    _as_seed_stack,
    _ba_patient_contributions,
    _ba_observed,
    _contribution_p_value,
    _tail_nll_patient_contributions,
    _tail_nll_observed,
)
from imbalance_benchmark.analysis.metrics import assign_tiers
from imbalance_benchmark.analysis.query import load_seed_predictions, load_test_identity
from imbalance_benchmark.common import split_paths

__all__ = [
    "crossed_block_permutation_ba",
    "crossed_block_permutation_tail_nll",
    "crossed_p_value",
    "load_freeze",
]

Block = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def _prepare(blocks: list[Block], rank: int) -> list[Block]:
    """Normalize every block to an explicit confirmation-seed axis."""
    return [
        (labels, _as_seed_stack(method, rank), _as_seed_stack(ce, rank), case_ids)
        for labels, method, ce, case_ids in blocks
    ]


def _crossed_contributions(
    prepared: list[Block],
    contribution_for_block: Callable[
        [int, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        tuple[np.ndarray, np.ndarray],
    ],
) -> np.ndarray:
    """Average split contributions while sharing one swap for repeated patients."""
    cases = np.unique(np.concatenate([block[3] for block in prepared]))
    contributions = np.zeros(len(cases), dtype=np.float64)
    for index, block in enumerate(prepared):
        block_cases, block_contributions = contribution_for_block(index, *block)
        contributions[np.searchsorted(cases, block_cases)] += block_contributions
    return contributions / len(prepared)


def crossed_block_permutation_ba(
    blocks: list[Block], n_classes: int, n_permutations: int = 100_000, seed: int = 0
) -> float:
    """Permute paired patient blocks while recomputing the equal-split BA mean."""
    prepared = _prepare(blocks, 1)
    contributions = _crossed_contributions(
        prepared,
        lambda _, labels, method, ce, case_ids: _ba_patient_contributions(
            labels, method, ce, case_ids, n_classes
        ),
    )
    observed = float(
        np.mean(
            [
                _ba_observed(labels, method, n_classes)
                - _ba_observed(labels, ce, n_classes)
                for labels, method, ce, _ in prepared
            ]
        )
    )
    contributions[-1] += observed - contributions.sum()
    return _contribution_p_value(contributions, observed, n_permutations, seed)


def _crossed_tail_observed(
    prepared: list[Block], split_tails: list[list[int]]
) -> float:
    return float(
        np.mean(
            [
                _tail_nll_observed(labels, ce, tails)
                - _tail_nll_observed(labels, method, tails)
                for (labels, method, ce, _), tails in zip(
                    prepared, split_tails, strict=True
                )
            ]
        )
    )


def crossed_block_permutation_tail_nll(
    blocks: list[Block],
    tail_classes: list[int] | list[list[int]],
    n_permutations: int = 100_000,
    seed: int = 0,
) -> float:
    """Permute paired blocks while recomputing the equal-split tail-NLL mean."""
    split_tails: list[list[int]]
    if tail_classes and isinstance(tail_classes[0], list):
        split_tails = cast(list[list[int]], tail_classes)
    else:
        split_tails = [cast(list[int], tail_classes)] * len(blocks)
    prepared = _prepare(blocks, 2)
    contributions = _crossed_contributions(
        prepared,
        lambda index, labels, method, ce, case_ids: _tail_nll_patient_contributions(
            labels, method, ce, case_ids, split_tails[index]
        ),
    )
    observed = _crossed_tail_observed(prepared, split_tails)
    contributions[-1] += observed - contributions.sum()
    return _contribution_p_value(contributions, observed, n_permutations, seed)


def load_freeze(paths: dict[str, Path]) -> dict[str, Any]:
    """Load the frozen analysis manifest, if `freeze` has already produced one."""
    freeze_path = paths["data"] / "manifest_freeze.json"
    return json.loads(freeze_path.read_text()) if freeze_path.exists() else {}


def _gate_eligible(entry: dict[str, Any]) -> bool:
    """Only the four confirmatory methods with a passed gate get a permutation p-value.

    Exploratory methods (§3.6) keep effects and CIs but no hypothesis test.
    """
    if entry["method"] == "ce" or not entry.get("gate_passed"):
        return False
    return entry["method"] in PRIMARY_METHODS


def _gate_blocks(
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


def _gate_tail_classes(
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


def crossed_p_value(
    entry: dict[str, Any],
    base_paths: dict[str, Path],
    config: dict[str, Any],
    seed: int,
) -> float | None:
    """Calculate the gate statistic's one shared-block permutation p-value across splits."""
    if not _gate_eligible(entry):
        return None
    is_mil = config.get("dataset", {}).get("regime", "patch") == "wsi"
    loaded = _gate_blocks(entry, base_paths, is_mil)
    if loaded is None:
        return None
    blocks, method_data = loaded
    class_names = method_data["class_names"]
    if entry["gate"] == "discrimination":
        return crossed_block_permutation_ba(blocks, len(class_names), seed=seed)
    tail_classes = _gate_tail_classes(entry, base_paths, class_names)
    return crossed_block_permutation_tail_nll(blocks, tail_classes, seed=seed)
