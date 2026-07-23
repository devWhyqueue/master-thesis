"""Permutation tests for the three fixed patient-split repetitions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np

from imbalance_benchmark.analysis.inference.holm import PRIMARY_METHODS
from imbalance_benchmark.analysis.inference.permutation import (
    _as_seed_stack,
    _expand_swap_to_rows,
    _p_value,
    _seed_mean_ba,
    _seed_mean_tail_nll,
    _swap_batches,
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


def _ba_statistics(
    prepared: list[Block],
    swaps: np.ndarray,
    row_indices: list[np.ndarray],
    classes: int,
) -> np.ndarray:
    """Evaluate equal-split BA differences for one batch of shared patient swaps."""
    values = []
    for (labels, method, ce, _), row_idx in zip(prepared, row_indices, strict=True):
        swap = _expand_swap_to_rows(swaps, row_idx)[None, :, :]
        first = np.where(swap, ce[:, :, None], method[:, :, None])
        second = np.where(swap, method[:, :, None], ce[:, :, None])
        values.append(
            _seed_mean_ba(labels, first, classes)
            - _seed_mean_ba(labels, second, classes)
        )
    return np.mean(values, axis=0)


def crossed_block_permutation_ba(
    blocks: list[Block], n_classes: int, n_permutations: int = 100_000, seed: int = 0
) -> float:
    """Permute paired patient blocks while recomputing the equal-split BA mean."""
    cases, prepared = (
        np.unique(np.concatenate([block[3] for block in blocks])),
        _prepare(blocks, 1),
    )
    observed = np.mean(
        [
            _seed_mean_ba(y, m[:, :, None], n_classes)[0]
            - _seed_mean_ba(y, c[:, :, None], n_classes)[0]
            for y, m, c, _ in prepared
        ]
    )
    row_indices = [np.searchsorted(cases, block[3]) for block in prepared]
    enumerated, batches = _swap_batches(len(cases), n_permutations, seed)
    statistics = [
        _ba_statistics(prepared, swaps, row_indices, n_classes) for swaps in batches
    ]
    return _p_value(float(observed), np.concatenate(statistics), enumerated)


def _nll_statistics(
    prepared: list[Block],
    swaps: np.ndarray,
    row_indices: list[np.ndarray],
    tails: list[list[int]],
) -> np.ndarray:
    """Evaluate equal-split tail-NLL differences for one batch of shared swaps."""
    values = []
    for (labels, method, ce, _), row_idx, split_tails in zip(
        prepared, row_indices, tails, strict=True
    ):
        swap = _expand_swap_to_rows(swaps, row_idx)[None, :, :, None]
        first = np.where(swap, ce[:, :, None, :], method[:, :, None, :])
        second = np.where(swap, method[:, :, None, :], ce[:, :, None, :])
        values.append(
            _seed_mean_tail_nll(labels, second, split_tails)
            - _seed_mean_tail_nll(labels, first, split_tails)
        )
    return np.mean(values, axis=0)


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
    cases, prepared = (
        np.unique(np.concatenate([block[3] for block in blocks])),
        _prepare(blocks, 2),
    )
    observed = np.mean(
        [
            _seed_mean_tail_nll(y, c[:, :, None, :], tails)[0]
            - _seed_mean_tail_nll(y, m[:, :, None, :], tails)[0]
            for (y, m, c, _), tails in zip(prepared, split_tails, strict=True)
        ]
    )
    row_indices = [np.searchsorted(cases, block[3]) for block in prepared]
    enumerated, batches = _swap_batches(len(cases), n_permutations, seed)
    statistics = [
        _nll_statistics(prepared, swaps, row_indices, split_tails) for swaps in batches
    ]
    return _p_value(float(observed), np.concatenate(statistics), enumerated)


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
