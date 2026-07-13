from __future__ import annotations

import itertools

import numpy as np

__all__ = [
    "paired_block_permutation_ba",
    "paired_block_permutation_tail_nll",
]

_ENUMERATE_PATIENT_LIMIT = 20
_BATCH_SIZE = 2000


def _swap_batches(
    n_patients: int, n_permutations: int, seed: int
) -> tuple[bool, list[np.ndarray]]:
    """Yield batches of a (n_patients, batch) boolean swap matrix.

    Enumerates all ``2**n_patients`` patient-block swaps when feasible
    (``n_patients <= 20``, matching the report's "all permutations are
    enumerated when feasible"); otherwise draws ``n_permutations`` random
    swaps for the plus-one-corrected Monte Carlo p-value. Batches keep peak
    memory bounded regardless of how many permutations are requested.
    """
    if n_patients <= _ENUMERATE_PATIENT_LIMIT:
        combos = np.array(
            list(itertools.product([False, True], repeat=n_patients)), dtype=bool
        ).T
        return True, [
            combos[:, i : i + _BATCH_SIZE]
            for i in range(0, combos.shape[1], _BATCH_SIZE)
        ]
    rng = np.random.default_rng(seed)
    batches = []
    remaining = n_permutations
    while remaining > 0:
        size = min(_BATCH_SIZE, remaining)
        batches.append(rng.random((n_patients, size)) < 0.5)
        remaining -= size
    return False, batches


def _expand_swap_to_rows(
    swap_patients: np.ndarray, unique_cases: np.ndarray, row_case_ids: np.ndarray
) -> np.ndarray:
    """Broadcast a (n_patients, batch) swap matrix to (n_rows, batch)."""
    position = {c: i for i, c in enumerate(unique_cases)}
    row_idx = np.asarray([position[c] for c in row_case_ids])
    return swap_patients[row_idx, :]


def _balanced_accuracy_batch(
    labels: np.ndarray, preds: np.ndarray, n_classes: int
) -> np.ndarray:
    """Balanced accuracy per permutation column for a (n_rows, batch) preds matrix."""
    batch = preds.shape[1]
    out = np.zeros(batch, dtype=np.float64)
    for c in range(n_classes):
        mask = labels == c
        if not mask.any():
            continue
        out += (preds[mask, :] == c).mean(axis=0)
    return out / n_classes


def _tail_nll_batch(
    labels: np.ndarray, probs_stack: np.ndarray, n_rows: int, tail_classes: list[int]
) -> np.ndarray:
    """Tail-group macro NLL per permutation column for a (n_rows, batch, n_classes) probs stack."""
    batch = probs_stack.shape[1]
    out = np.zeros(batch, dtype=np.float64)
    counted = 0
    for c in tail_classes:
        mask = labels == c
        if not mask.any():
            continue
        p_true = np.clip(probs_stack[mask, :, c], 1e-12, 1.0)
        out += -np.log(p_true).mean(axis=0)
        counted += 1
    return out / max(counted, 1)


def _p_value(observed: float, extremes: np.ndarray, enumerated: bool) -> float:
    n = len(extremes)
    exceed = int(np.sum(np.abs(extremes) >= abs(observed)))
    if enumerated:
        return exceed / n
    return (exceed + 1) / (n + 1)


def _as_seed_stack(values: np.ndarray, prediction_rank: int) -> np.ndarray:
    """Normalize one prediction array or matched seed stack to ``(seed, row, ...)``."""
    if values.ndim == prediction_rank:
        return values[None, ...]
    if values.ndim == prediction_rank + 1:
        return values
    raise ValueError("Unexpected prediction-array rank for permutation test")


def _seed_mean_ba(labels: np.ndarray, preds: np.ndarray, n_classes: int) -> np.ndarray:
    """Compute one balanced-accuracy value per permutation after seed averaging."""
    return np.mean(
        [
            _balanced_accuracy_batch(labels, seed_preds, n_classes)
            for seed_preds in preds
        ],
        axis=0,
    )


def _seed_mean_tail_nll(
    labels: np.ndarray, probs: np.ndarray, tail_classes: list[int]
) -> np.ndarray:
    """Compute tail NLL per permutation after averaging matched seed blocks."""
    return np.mean(
        [
            _tail_nll_batch(labels, seed_probs, len(labels), tail_classes)
            for seed_probs in probs
        ],
        axis=0,
    )


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

    Under each permutation the complete method/CE prediction blocks are
    swapped together within a patient (all of that patient's rows), and the
    balanced-accuracy difference is recomputed; two-sided p-value with the
    plus-one correction when permutations are sampled rather than enumerated.
    """
    method_stack, ce_stack = (
        _as_seed_stack(method_preds, 1),
        _as_seed_stack(ce_preds, 1),
    )
    if method_stack.shape != ce_stack.shape:
        raise ValueError("Permutation pairs require equal seed and prediction shapes")
    observed = float(
        _seed_mean_ba(labels, method_stack[:, :, None], n_classes)[0]
        - _seed_mean_ba(labels, ce_stack[:, :, None], n_classes)[0]
    )
    unique_cases = np.unique(case_ids)
    enumerated, batches = _swap_batches(len(unique_cases), n_permutations, seed)
    stats = []
    for swap_patients in batches:
        swap_rows = _expand_swap_to_rows(swap_patients, unique_cases, case_ids)
        swap = swap_rows[None, :, :]
        preds_a = np.where(swap, ce_stack[:, :, None], method_stack[:, :, None])
        preds_b = np.where(swap, method_stack[:, :, None], ce_stack[:, :, None])
        stats.append(
            _seed_mean_ba(labels, preds_a, n_classes)
            - _seed_mean_ba(labels, preds_b, n_classes)
        )
    return _p_value(float(observed), np.concatenate(stats), enumerated)


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
    observed = float(
        _seed_mean_tail_nll(labels, ce_stack[:, :, None, :], tail_classes)[0]
        - _seed_mean_tail_nll(labels, method_stack[:, :, None, :], tail_classes)[0]
    )
    unique_cases = np.unique(case_ids)
    enumerated, batches = _swap_batches(len(unique_cases), n_permutations, seed)
    stats = []
    for swap_patients in batches:
        swap_rows = _expand_swap_to_rows(swap_patients, unique_cases, case_ids)
        swap3 = swap_rows[None, :, :, None]
        probs_a = np.where(swap3, ce_stack[:, :, None, :], method_stack[:, :, None, :])
        probs_b = np.where(swap3, method_stack[:, :, None, :], ce_stack[:, :, None, :])
        nll_a = _seed_mean_tail_nll(labels, probs_a, tail_classes)
        nll_b = _seed_mean_tail_nll(labels, probs_b, tail_classes)
        stats.append(nll_b - nll_a)
    return _p_value(float(observed), np.concatenate(stats), enumerated)
