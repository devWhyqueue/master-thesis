from __future__ import annotations

import itertools
from collections.abc import Iterator

import numpy as np

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


def _ba_observed(labels: np.ndarray, predictions: np.ndarray, n_classes: int) -> float:
    values = []
    for seed_predictions in predictions:
        value = 0.0
        for class_index in range(n_classes):
            mask = labels == class_index
            if mask.any():
                value += (seed_predictions[mask] == class_index).mean()
        values.append(value / n_classes)
    return float(np.mean(values))


def _tail_nll_observed(
    labels: np.ndarray, probabilities: np.ndarray, tail_classes: list[int]
) -> float:
    values = []
    for seed_probabilities in probabilities:
        value = 0.0
        counted = 0
        for class_index in tail_classes:
            mask = labels == class_index
            if mask.any():
                value += -np.log(
                    np.clip(seed_probabilities[mask, class_index], 1e-12, 1.0)
                ).mean()
                counted += 1
        values.append(value / max(counted, 1))
    return float(np.mean(values))


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


def _ba_patient_contributions(
    labels: np.ndarray,
    method_preds: np.ndarray,
    ce_preds: np.ndarray,
    case_ids: np.ndarray,
    n_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return additive patient contributions to matched seed-mean BA difference."""
    rows = np.zeros(len(labels), dtype=np.float64)
    for class_index in range(n_classes):
        mask = labels == class_index
        if not mask.any():
            continue
        method_correct = method_preds[:, mask] == class_index
        ce_correct = ce_preds[:, mask] == class_index
        rows[mask] = (method_correct.astype(float) - ce_correct).mean(axis=0) / (
            n_classes * mask.sum()
        )
    return _patient_contributions(case_ids, rows)


def _tail_nll_patient_contributions(
    labels: np.ndarray,
    method_probs: np.ndarray,
    ce_probs: np.ndarray,
    case_ids: np.ndarray,
    tail_classes: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Return additive patient contributions to matched seed-mean tail-NLL difference."""
    rows = np.zeros(len(labels), dtype=np.float64)
    present_tails = [
        class_index for class_index in tail_classes if (labels == class_index).any()
    ]
    for class_index in present_tails:
        mask = labels == class_index
        method_nll = -np.log(np.clip(method_probs[:, mask, class_index], 1e-12, 1.0))
        ce_nll = -np.log(np.clip(ce_probs[:, mask, class_index], 1e-12, 1.0))
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
