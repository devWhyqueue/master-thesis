"""Unit tests for the covariance-trace ICC estimator and interpretation classifier."""

from __future__ import annotations

import numpy as np
import pytest
from imbalance_benchmark.analysis.predictors.signals.icc import icc_estimate

from breadth.icc import sample_class_indices, sample_class_scores
from redundancy.analyze import classify
from redundancy.estimator import cluster_stats, weighted_icc


def _clustered_scalars(seed: int, n_cases: int = 8, per_case: int = 15):
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 2.0, size=n_cases)
    scores, case_ids = [], []
    for i in range(n_cases):
        scores.extend(centers[i] + rng.normal(0, 1.0, size=per_case))
        case_ids.extend([f"c{i}"] * per_case)
    return np.array(scores), np.array(case_ids)


def _clustered_vectors(seed: int, n_cases: int = 6, per_case: int = 10, dim: int = 5):
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 2.0, size=(n_cases, dim))
    case_ids = np.repeat([f"c{i}" for i in range(n_cases)], per_case)
    features = np.vstack(
        [centers[i] + rng.normal(0, 0.5, size=(per_case, dim)) for i in range(n_cases)]
    )
    return features, case_ids


def test_weighted_icc_unit_weights_matches_scalar_icc_estimate():
    """Unit weights on a scalar column reproduce exp-2's one-way ANOVA estimator."""
    scores, case_ids = _clustered_scalars(seed=0)
    stats = cluster_stats(scores[:, np.newaxis], case_ids)
    weights = np.ones((1, len(stats.cases)))

    got = weighted_icc(stats, weights)[0]
    expected = icc_estimate(scores, case_ids)

    assert expected is not None
    assert 0.0 < expected < 1.0  # away from the clip boundary
    assert np.isclose(got, expected, atol=1e-8)


def test_weighted_icc_integer_weights_equal_duplicated_clusters():
    """A weight of k on one patient equals k literal copies of that patient's rows."""
    features, case_ids = _clustered_vectors(seed=1, n_cases=3, per_case=8, dim=3)
    stats = cluster_stats(features, case_ids)
    weights = np.array([[2.0 if case == "c0" else 1.0 for case in stats.cases]])
    got = weighted_icc(stats, weights)[0]

    dup_mask = case_ids == "c0"
    dup_features = np.concatenate([features, features[dup_mask]], axis=0)
    dup_case_ids = np.concatenate(
        [case_ids, np.full(int(dup_mask.sum()), "c0_copy")]
    )
    dup_stats = cluster_stats(dup_features, dup_case_ids)
    expected = weighted_icc(dup_stats, np.ones((1, len(dup_stats.cases))))[0]

    assert np.isclose(got, expected, atol=1e-8)


def test_full_icc_invariant_to_orthogonal_rotation():
    """Rotating the feature basis leaves the covariance-trace ICC unchanged."""
    features, case_ids = _clustered_vectors(seed=2)
    stats = cluster_stats(features, case_ids)
    weights = np.ones((1, len(stats.cases)))
    rho = weighted_icc(stats, weights)[0]

    rng = np.random.default_rng(3)
    q, _ = np.linalg.qr(rng.normal(size=(features.shape[1], features.shape[1])))
    rotated_stats = cluster_stats(features @ q, case_ids)
    rho_rotated = weighted_icc(rotated_stats, weights)[0]

    assert np.isclose(rho, rho_rotated, atol=1e-8)


def test_weighted_icc_raises_on_too_few_patients():
    """A single patient (H<2) is an undefined estimate, not a silent zero."""
    features = np.array([[1.0], [2.0], [3.0]])
    case_ids = np.array(["a", "a", "a"])
    stats = cluster_stats(features, case_ids)
    with pytest.raises(ValueError):
        weighted_icc(stats, np.ones((1, 1)))


def test_weighted_icc_raises_on_no_within_patient_degrees_of_freedom():
    """One patch per patient (N == H) leaves no within-patient variance."""
    features = np.array([[1.0], [2.0], [3.0]])
    case_ids = np.array(["a", "b", "c"])
    stats = cluster_stats(features, case_ids)
    with pytest.raises(ValueError):
        weighted_icc(stats, np.ones((1, 3)))


@pytest.mark.parametrize(
    ("b_ci", "res_full", "res_single", "expected"),
    [
        ((-0.5, 0.5), 0.2, 0.25, "underestimated_redundancy"),
        ((-0.5, 0.5), 0.3, 0.25, "inconclusive"),  # fits worse than the single measure
        ((1.5, 2.0), 0.5, 0.25, "breadth_beyond_redundancy"),
        ((0.5, 1.5), 0.2, 0.25, "inconclusive"),  # interval straddles the threshold
        ((-2.0, 0.5), 0.2, 0.25, "inconclusive"),  # interval extends below -1
    ],
)
def test_classify_truth_table(b_ci, res_full, res_single, expected):
    assert classify(b_ci, res_full, res_single) == expected


def test_sample_class_indices_matches_sample_class_scores():
    """The exp-5 refactor keeps sample_class_scores behavior-identical."""
    case_ids = np.array([f"c{i // 3}" for i in range(30)])
    features = np.random.default_rng(6).normal(size=(30, 4))
    direction = np.array([1.0, 0.0, 0.0, 0.0])

    rng_direct = np.random.default_rng(5)
    scores_direct, cases_direct = sample_class_scores(
        features, case_ids, direction, rng_direct, case_cap=5, patch_cap=2
    )

    rng_indices = np.random.default_rng(5)
    indices, cases_from_indices = sample_class_indices(
        case_ids, rng_indices, case_cap=5, patch_cap=2
    )

    assert np.array_equal(cases_direct, cases_from_indices)
    assert np.allclose(scores_direct, features[indices] @ direction)
