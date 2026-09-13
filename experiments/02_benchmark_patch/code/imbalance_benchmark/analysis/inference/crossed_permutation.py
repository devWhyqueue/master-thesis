"""Permutation tests for the three fixed patient-split repetitions."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Callable
from typing import Any, cast

import numpy as np

from imbalance_benchmark.analysis.inference.confirmatory.gate_blocks import (
    Arm,
    Block,
    gate_blocks,
    gate_tail_classes,
    load_freeze,
)
from imbalance_benchmark.analysis.inference.confirmatory.holm import (
    MATCHED_CONTRAST_METHOD,
    PRIMARY_METHODS,
)
from imbalance_benchmark.analysis.inference.confirmatory.arms import (
    _as_members,
    _ba_observed,
    _tail_nll_observed,
)
from imbalance_benchmark.analysis.inference.permutation import (
    _as_seed_stack,
    _ba_patient_contributions,
    _contribution_p_value,
    _tail_nll_patient_contributions,
)

__all__ = [
    "crossed_block_permutation_ba",
    "crossed_block_permutation_tail_nll",
    "crossed_p_value",
    "load_freeze",
]


def _prepare(blocks: list[Block], rank: int) -> list[Block]:
    """Normalize every block's arms to lists of explicit confirmation-seed stacks.

    Each arm may be a bare array (one confirmatory member, e.g. one mitigation
    method) or a list of several (protocol app:testing's matched-vs-unmatched
    contrast, where either side can average more than one method).
    """
    return [
        (
            labels,
            [_as_seed_stack(member, rank) for member in _as_members(method)],
            [_as_seed_stack(member, rank) for member in _as_members(ce)],
            case_ids,
        )
        for labels, method, ce, case_ids in blocks
    ]


def _crossed_contributions(
    prepared: list[Block],
    contribution_for_block: Callable[
        [int, np.ndarray, Arm, Arm, np.ndarray],
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
                _ba_observed(labels, method, case_ids, n_classes)
                - _ba_observed(labels, ce, case_ids, n_classes)
                for labels, method, ce, case_ids in prepared
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
                _tail_nll_observed(labels, ce, case_ids, tails)
                - _tail_nll_observed(labels, method, case_ids, tails)
                for (labels, method, ce, case_ids), tails in zip(
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


def _gate_eligible(entry: dict[str, Any]) -> bool:
    """Only the confirmatory methods and the matched contrast, gate-passing, get a p-value.

    Exploratory methods (§3.6) keep effects and CIs but no hypothesis test.
    """
    if entry["method"] == "ce" or not entry.get("gate_passed"):
        return False
    return (
        entry["method"] in PRIMARY_METHODS or entry["method"] == MATCHED_CONTRAST_METHOD
    )


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
    loaded = gate_blocks(entry, base_paths, is_mil)
    if loaded is None:
        return None
    blocks, method_data = loaded
    class_names = method_data["class_names"]
    if entry["gate"] == "discrimination":
        return crossed_block_permutation_ba(blocks, len(class_names), seed=seed)
    tail_classes = gate_tail_classes(entry, base_paths, class_names)
    return crossed_block_permutation_tail_nll(blocks, tail_classes, seed=seed)
