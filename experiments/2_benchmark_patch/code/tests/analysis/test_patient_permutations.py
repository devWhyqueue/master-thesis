from __future__ import annotations

import itertools

import numpy as np
import pytest

from imbalance_benchmark.analysis.inference.crossed_permutation import (
    crossed_block_permutation_ba,
    crossed_block_permutation_tail_nll,
)
from imbalance_benchmark.analysis.inference.permutation import (
    paired_block_permutation_ba,
    paired_block_permutation_tail_nll,
)
from imbalance_benchmark.analysis.inference.context import BootstrapContext


def _brute_swaps(
    n_patients: int, n_permutations: int, seed: int
) -> tuple[bool, np.ndarray]:
    if n_patients <= 20:
        return True, np.asarray(
            list(itertools.product([False, True], repeat=n_patients))
        ).T
    return False, np.random.default_rng(seed).random((n_patients, n_permutations)) < 0.5


def _p_value(observed: float, statistics: np.ndarray, enumerated: bool) -> float:
    magnitude = np.abs(statistics)
    threshold = abs(observed)
    exceed = np.count_nonzero(
        (magnitude >= threshold)
        | np.isclose(magnitude, threshold, rtol=1e-12, atol=1e-15)
    )
    return (
        exceed / len(statistics) if enumerated else (exceed + 1) / (len(statistics) + 1)
    )


def _brute_ba(
    labels: np.ndarray,
    method: np.ndarray,
    ce: np.ndarray,
    case_ids: np.ndarray,
    classes: int,
    n_permutations: int,
    seed: int,
) -> float:
    method = method[None] if method.ndim == 1 else method
    ce = ce[None] if ce.ndim == 1 else ce
    cases, inverse = np.unique(case_ids, return_inverse=True)
    enumerated, swaps = _brute_swaps(len(cases), n_permutations, seed)
    statistics = []
    for swap in swaps.T:
        rows = swap[inverse]
        first = np.where(rows, ce, method)
        second = np.where(rows, method, ce)
        per_seed = []
        for first_seed, second_seed in zip(first, second, strict=True):
            first_ba = np.mean(
                [(first_seed[labels == cls] == cls).mean() for cls in range(classes)]
            )
            second_ba = np.mean(
                [(second_seed[labels == cls] == cls).mean() for cls in range(classes)]
            )
            per_seed.append(first_ba - second_ba)
        statistics.append(np.mean(per_seed))
    observed = np.mean(
        [
            np.mean(
                [(seed_preds[labels == cls] == cls).mean() for cls in range(classes)]
            )
            for seed_preds in method
        ]
    ) - np.mean(
        [
            np.mean(
                [(seed_preds[labels == cls] == cls).mean() for cls in range(classes)]
            )
            for seed_preds in ce
        ]
    )
    return _p_value(observed, np.asarray(statistics), enumerated)


def _brute_tail_nll(
    labels: np.ndarray,
    method: np.ndarray,
    ce: np.ndarray,
    case_ids: np.ndarray,
    tails: list[int],
    n_permutations: int,
    seed: int,
) -> float:
    method = method[None] if method.ndim == 2 else method
    ce = ce[None] if ce.ndim == 2 else ce
    cases, inverse = np.unique(case_ids, return_inverse=True)
    enumerated, swaps = _brute_swaps(len(cases), n_permutations, seed)

    def statistic(first: np.ndarray, second: np.ndarray) -> float:
        values = []
        for first_seed, second_seed in zip(first, second, strict=True):
            first_nll = np.mean(
                [
                    -np.log(np.clip(first_seed[labels == cls, cls], 1e-12, 1.0)).mean()
                    for cls in tails
                ]
            )
            second_nll = np.mean(
                [
                    -np.log(np.clip(second_seed[labels == cls, cls], 1e-12, 1.0)).mean()
                    for cls in tails
                ]
            )
            values.append(second_nll - first_nll)
        return float(np.mean(values))

    statistics = []
    for swap in swaps.T:
        rows = swap[inverse]
        statistics.append(
            statistic(
                np.where(rows[None, :, None], ce, method),
                np.where(rows[None, :, None], method, ce),
            )
        )
    return _p_value(statistic(ce, method), np.asarray(statistics), enumerated)


def _brute_crossed_ba(
    blocks: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    classes: int,
    n_permutations: int,
    seed: int,
) -> float:
    cases = np.unique(np.concatenate([block[3] for block in blocks]))
    enumerated, swaps = _brute_swaps(len(cases), n_permutations, seed)
    statistics = []
    for swap in swaps.T:
        values = []
        for labels, method, ce, case_ids in blocks:
            method = method[None] if method.ndim == 1 else method
            ce = ce[None] if ce.ndim == 1 else ce
            rows = swap[np.searchsorted(cases, case_ids)]
            first = np.where(rows, ce, method)
            second = np.where(rows, method, ce)
            values.append(
                np.mean(
                    [
                        np.mean(
                            [
                                (first_seed[labels == cls] == cls).mean()
                                - (second_seed[labels == cls] == cls).mean()
                                for cls in range(classes)
                            ]
                        )
                        for first_seed, second_seed in zip(first, second, strict=True)
                    ]
                )
            )
        statistics.append(np.mean(values))
    # The equal-split statistic comes from the unpermuted arrays, not p-values.
    observed_statistics = []
    for labels, method, ce, _ in blocks:
        method = method[None] if method.ndim == 1 else method
        ce = ce[None] if ce.ndim == 1 else ce
        observed_statistics.append(
            np.mean(
                [
                    np.mean(
                        [
                            (method_seed[labels == cls] == cls).mean()
                            - (ce_seed[labels == cls] == cls).mean()
                            for cls in range(classes)
                        ]
                    )
                    for method_seed, ce_seed in zip(method, ce, strict=True)
                ]
            )
        )
    return _p_value(np.mean(observed_statistics), np.asarray(statistics), enumerated)


@pytest.mark.parametrize("seeds", [1, 5])
def test_patient_contribution_permutations_match_brute_enumeration(seeds: int) -> None:
    labels = np.array([0, 0, 1, 1, 2, 2, 2, 0])
    case_ids = np.array(["p0", "p0", "p1", "p1", "p2", "p2", "p3", "p3"])
    rng = np.random.default_rng(4)
    method_preds = rng.integers(0, 3, size=(seeds, len(labels)))
    ce_preds = rng.integers(0, 3, size=(seeds, len(labels)))
    method_probs = rng.dirichlet(np.ones(3), size=(seeds, len(labels)))
    ce_probs = rng.dirichlet(np.ones(3), size=(seeds, len(labels)))
    if seeds == 1:
        method_preds, ce_preds = method_preds[0], ce_preds[0]
        method_probs, ce_probs = method_probs[0], ce_probs[0]

    assert paired_block_permutation_ba(
        labels, method_preds, ce_preds, case_ids, 3, n_permutations=17, seed=9
    ) == pytest.approx(_brute_ba(labels, method_preds, ce_preds, case_ids, 3, 17, 9))
    assert paired_block_permutation_tail_nll(
        labels, method_probs, ce_probs, case_ids, [1, 2], n_permutations=17, seed=9
    ) == pytest.approx(
        _brute_tail_nll(labels, method_probs, ce_probs, case_ids, [1, 2], 17, 9)
    )


def test_patient_contribution_permutations_match_brute_monte_carlo() -> None:
    labels = np.tile([0, 1, 2], 21)
    case_ids = np.repeat(np.arange(21).astype(str), 3)
    rng = np.random.default_rng(7)
    method = rng.integers(0, 3, size=(5, len(labels)))
    ce = rng.integers(0, 3, size=(5, len(labels)))
    method_probs = rng.dirichlet(np.ones(3), size=(5, len(labels)))
    ce_probs = rng.dirichlet(np.ones(3), size=(5, len(labels)))

    assert paired_block_permutation_ba(
        labels, method, ce, case_ids, 3, n_permutations=113, seed=2
    ) == pytest.approx(_brute_ba(labels, method, ce, case_ids, 3, 113, 2))
    assert paired_block_permutation_tail_nll(
        labels, method_probs, ce_probs, case_ids, [0, 2], n_permutations=113, seed=2
    ) == pytest.approx(
        _brute_tail_nll(labels, method_probs, ce_probs, case_ids, [0, 2], 113, 2)
    )


def test_crossed_patient_contributions_share_repeated_patient_swaps() -> None:
    labels = np.array([0, 0, 1, 1, 2, 2])
    patients = np.array(["p0", "p0", "p1", "p1", "p2", "p2"])
    rng = np.random.default_rng(11)
    blocks = []
    for _ in range(3):
        blocks.append(
            (
                labels,
                rng.integers(0, 3, size=(5, len(labels))),
                rng.integers(0, 3, size=(5, len(labels))),
                patients,
            )
        )
    expected = _brute_crossed_ba(blocks, 3, 19, 5)
    assert crossed_block_permutation_ba(
        blocks, 3, n_permutations=19, seed=5
    ) == pytest.approx(expected)

    probability_block = (
        labels,
        rng.dirichlet(np.ones(3), size=(5, len(labels))),
        rng.dirichlet(np.ones(3), size=(5, len(labels))),
        patients,
    )
    probability_blocks = [probability_block] * 3
    expected_tail = _brute_tail_nll(*probability_block, [1, 2], 19, 5)
    assert crossed_block_permutation_tail_nll(
        probability_blocks, [1, 2], n_permutations=19, seed=5
    ) == pytest.approx(expected_tail)


def test_probability_only_secondary_distributions_match_full_outputs() -> None:
    context = object.__new__(BootstrapContext)
    context.case_ids = np.array(["p0", "p0", "p1", "p1"])
    context.slide_ids = np.array(["s0", "s1", "s2", "s3"])
    context.weights = __import__(
        "imbalance_benchmark.analysis.inference.bootstrap", fromlist=["PatientWeights"]
    ).PatientWeights(np.array([0, 0, 1, 1]), np.ones((2, 3)))
    context.n_replicates, context._seed, context._seed_indices = 3, 1, {}
    labels = np.array([0, 0, 1, 1])
    predictions = np.array([[0, 1, 1, 0], [0, 0, 1, 1]])
    probabilities = np.array(
        [
            [[0.9, 0.1], [0.4, 0.6], [0.2, 0.8], [0.7, 0.3]],
            [[0.8, 0.2], [0.6, 0.4], [0.1, 0.9], [0.3, 0.7]],
        ]
    )
    full = context.secondary_distributions(
        labels,
        predictions,
        probabilities,
        ["head", "tail"],
        {"head": "head", "tail": "tail"},
    )
    probability_only = context.probability_secondary_distributions(
        labels, probabilities, ["head", "tail"], {"head": "head", "tail": "tail"}
    )
    expected_keys = {
        key
        for key in full
        if key
        in {
            "negative_log_likelihood",
            "macro_nll",
            "brier_score",
            "expected_calibration_error",
        }
        or key.startswith(("nll:", "brier:", "tier_nll:", "tier_brier:"))
        or key.endswith(("_macro_nll", "_macro_brier"))
    }
    assert set(probability_only) == expected_keys
    for key in expected_keys:
        np.testing.assert_allclose(probability_only[key], full[key])
