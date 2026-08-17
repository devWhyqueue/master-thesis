from __future__ import annotations

import itertools
from collections.abc import Iterator

import numpy as np

from imbalance_benchmark.analysis.inference.confirmatory.arms import (
    _as_members,
    _ba_observed,
    _tail_nll_observed,
)

__all__ = [
    "paired_block_permutation_ba",
    "paired_block_permutation_tail_nll",
]

_ENUMERATE_PATIENT_LIMIT = 20
_BATCH_SIZE = 2000


def _swap_batches(
    n_patients: int, n_permutations: int, seed: int
) -> tuple[bool, Iterator[np.ndarray]]:
    """Stream patient-block swap batches without retaining permutation matrices."""
    if n_patients <= _ENUMERATE_PATIENT_LIMIT:
        return True, _enumerated_swap_batches(n_patients)
    return False, _random_swap_batches(n_patients, n_permutations, seed)


def _enumerated_swap_batches(n_patients: int) -> Iterator[np.ndarray]:
    combinations = itertools.product([False, True], repeat=n_patients)
    while batch := list(itertools.islice(combinations, _BATCH_SIZE)):
        yield np.asarray(batch, dtype=bool).T


def _random_swap_batches(
    n_patients: int, n_permutations: int, seed: int
) -> Iterator[np.ndarray]:
    rng = np.random.default_rng(seed)
    remaining = n_permutations
    while remaining:
        size = min(_BATCH_SIZE, remaining)
        yield rng.random((n_patients, size)) < 0.5
        remaining -= size


def _p_value_from_counts(exceed: int, total: int, enumerated: bool) -> float:
    if enumerated:
        return exceed / total
    return (exceed + 1) / (total + 1)


def _contribution_p_value(
    contributions: np.ndarray, observed: float, n_permutations: int, seed: int
) -> float:
    """Test a statistic represented by additive paired patient contributions."""
    enumerated, batches = _swap_batches(len(contributions), n_permutations, seed)
    exceed = total = 0
    for swaps in batches:
        statistics = observed - 2.0 * contributions @ swaps
        magnitude = np.abs(statistics)
        threshold = abs(observed)
        exceed += np.count_nonzero(
            (magnitude >= threshold)
            | np.isclose(magnitude, threshold, rtol=1e-12, atol=1e-15)
        )
        total += statistics.size
    return _p_value_from_counts(exceed, total, enumerated)


def _as_seed_stack(values: np.ndarray, prediction_rank: int) -> np.ndarray:
    """Normalize one prediction array or matched seed stack to ``(seed, row, ...)``."""
    if values.ndim == prediction_rank:
        return values[None, ...]
    if values.ndim == prediction_rank + 1:
        return values
    raise ValueError("Unexpected prediction-array rank for permutation test")


def _patient_contributions(
    case_ids: np.ndarray, row_contributions: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Sum exact row-statistic contributions into one contribution per patient."""
    cases, inverse = np.unique(case_ids, return_inverse=True)
    contributions = np.zeros(len(cases), dtype=np.float64)
    np.add.at(contributions, inverse, row_contributions)
    return cases, contributions


def _correctness_mean(
    members: list[np.ndarray], mask: np.ndarray, class_index: int
) -> np.ndarray:
    """Mean per-seed correctness across confirmatory members, shape ``(seed, row)``."""
    return np.mean(
        [_as_seed_stack(member, 1)[:, mask] == class_index for member in members],
        axis=0,
    )


def _ba_patient_contributions(
    labels: np.ndarray,
    method_preds: np.ndarray | list[np.ndarray],
    ce_preds: np.ndarray | list[np.ndarray],
    case_ids: np.ndarray,
    n_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return additive patient contributions to matched seed-mean BA difference.

    Either arm may be several confirmatory members (protocol app:testing's
    matched-vs-unmatched contrast); their per-row correctness is averaged
    before the paired per-seed difference, exact because BA is linear in
    per-row correctness.
    """
    method_members, ce_members = _as_members(method_preds), _as_members(ce_preds)
    rows = np.zeros(len(labels), dtype=np.float64)
    for class_index in range(n_classes):
        mask = labels == class_index
        if not mask.any():
            continue
        method_correct = _correctness_mean(method_members, mask, class_index)
        ce_correct = _correctness_mean(ce_members, mask, class_index)
        rows[mask] = (method_correct - ce_correct).mean(axis=0) / (
            n_classes * mask.sum()
        )
    return _patient_contributions(case_ids, rows)


def _nll_mean(
    members: list[np.ndarray], mask: np.ndarray, class_index: int
) -> np.ndarray:
    """Mean per-seed NLL across confirmatory members, shape ``(seed, row)``."""
    return np.mean(
        [
            -np.log(
                np.clip(_as_seed_stack(member, 2)[:, mask, class_index], 1e-12, 1.0)
            )
            for member in members
        ],
        axis=0,
    )


def _tail_nll_patient_contributions(
    labels: np.ndarray,
    method_probs: np.ndarray | list[np.ndarray],
    ce_probs: np.ndarray | list[np.ndarray],
    case_ids: np.ndarray,
    tail_classes: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Return additive patient contributions to matched seed-mean tail-NLL difference.

    Either arm may be several confirmatory members, averaged per row before
    the paired per-seed difference (see :func:`_ba_patient_contributions`).
    """
    method_members, ce_members = _as_members(method_probs), _as_members(ce_probs)
    rows = np.zeros(len(labels), dtype=np.float64)
    present_tails = [
        class_index for class_index in tail_classes if (labels == class_index).any()
    ]
    for class_index in present_tails:
        mask = labels == class_index
        method_nll = _nll_mean(method_members, mask, class_index)
        ce_nll = _nll_mean(ce_members, mask, class_index)
        rows[mask] = (ce_nll - method_nll).mean(axis=0) / (
            len(present_tails) * mask.sum()
        )
    return _patient_contributions(case_ids, rows)


def paired_block_permutation_ba(
    labels: np.ndarray,
    method_preds: np.ndarray,
    ce_preds: np.ndarray,
    case_ids: np.ndarray,
    n_classes: int,
    n_permutations: int = 100_000,
    seed: int = 0,
) -> float:
    """Paired patient-block permutation test for the discrimination-axis statistic.

    Swaps method/CE blocks within a patient and recomputes the balanced-
    accuracy difference; two-sided p-value, plus-one corrected when sampled.
    """
    method_stack, ce_stack = (
        _as_seed_stack(method_preds, 1),
        _as_seed_stack(ce_preds, 1),
    )
    if method_stack.shape != ce_stack.shape:
        raise ValueError("Permutation pairs require equal seed and prediction shapes")
    _, contributions = _ba_patient_contributions(
        labels, method_stack, ce_stack, case_ids, n_classes
    )
    observed = _ba_observed(labels, method_stack, n_classes) - _ba_observed(
        labels, ce_stack, n_classes
    )
    contributions[-1] += observed - contributions.sum()
    return _contribution_p_value(contributions, observed, n_permutations, seed)


def paired_block_permutation_tail_nll(
    labels: np.ndarray,
    method_probs: np.ndarray,
    ce_probs: np.ndarray,
    case_ids: np.ndarray,
    tail_classes: list[int],
    n_permutations: int = 100_000,
    seed: int = 0,
) -> float:
    """Paired patient-block permutation test for the calibration-axis statistic.

    Statistic is oriented so positive means improvement:
    ``NLL_tail(CE) - NLL_tail(method)``.
    """
    method_stack, ce_stack = (
        _as_seed_stack(method_probs, 2),
        _as_seed_stack(ce_probs, 2),
    )
    if method_stack.shape != ce_stack.shape:
        raise ValueError("Permutation pairs require equal seed and prediction shapes")
    _, contributions = _tail_nll_patient_contributions(
        labels, method_stack, ce_stack, case_ids, tail_classes
    )
    observed = _tail_nll_observed(labels, ce_stack, tail_classes) - _tail_nll_observed(
        labels, method_stack, tail_classes
    )
    contributions[-1] += observed - contributions.sum()
    return _contribution_p_value(contributions, observed, n_permutations, seed)
