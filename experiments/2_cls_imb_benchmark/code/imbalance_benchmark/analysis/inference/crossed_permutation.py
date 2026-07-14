"""Permutation tests for the three fixed patient-split repetitions."""

from __future__ import annotations

import numpy as np
from typing import cast

from imbalance_benchmark.analysis.inference.permutation import (
    _as_seed_stack,
    _expand_swap_to_rows,
    _p_value,
    _seed_mean_ba,
    _seed_mean_tail_nll,
    _swap_batches,
)

__all__ = ["crossed_block_permutation_ba", "crossed_block_permutation_tail_nll"]

Block = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def _prepare(blocks: list[Block], rank: int) -> list[Block]:
    """Normalize every block to an explicit confirmation-seed axis."""
    return [
        (labels, _as_seed_stack(method, rank), _as_seed_stack(ce, rank), case_ids)
        for labels, method, ce, case_ids in blocks
    ]


def _ba_statistics(
    prepared: list[Block], swaps: np.ndarray, cases: np.ndarray, classes: int
) -> np.ndarray:
    """Evaluate equal-split BA differences for one batch of shared patient swaps."""
    values = []
    for labels, method, ce, case_ids in prepared:
        swap = _expand_swap_to_rows(swaps, cases, case_ids)[None, :, :]
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
    enumerated, batches = _swap_batches(len(cases), n_permutations, seed)
    statistics = [
        _ba_statistics(prepared, swaps, cases, n_classes) for swaps in batches
    ]
    return _p_value(float(observed), np.concatenate(statistics), enumerated)


def _nll_statistics(
    prepared: list[Block], swaps: np.ndarray, cases: np.ndarray, tails: list[list[int]]
) -> np.ndarray:
    """Evaluate equal-split tail-NLL differences for one batch of shared swaps."""
    values = []
    for (labels, method, ce, case_ids), split_tails in zip(
        prepared, tails, strict=True
    ):
        swap = _expand_swap_to_rows(swaps, cases, case_ids)[None, :, :, None]
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
    enumerated, batches = _swap_batches(len(cases), n_permutations, seed)
    statistics = [
        _nll_statistics(prepared, swaps, cases, split_tails) for swaps in batches
    ]
    return _p_value(float(observed), np.concatenate(statistics), enumerated)
