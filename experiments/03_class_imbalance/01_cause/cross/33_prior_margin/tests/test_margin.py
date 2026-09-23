"""Unit tests for exp-33's prior-injection and subset-renormalisation logic."""

from __future__ import annotations

import numpy as np
import pytest

from margin.inject import injected_preds, rho_shares
from margin.subset_eval import renormalize_subset


def _random_probs(rng: np.random.Generator, n: int, k: int) -> np.ndarray:
    logits = rng.normal(size=(n, k))
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    return probs / probs.sum(axis=1, keepdims=True)


def test_uniform_prior_leaves_preds_unchanged() -> None:
    """Injecting a uniform (rho=1) share profile does not move any argmax."""
    rng = np.random.default_rng(0)
    probs = _random_probs(rng, 200, 7)
    expected = np.argmax(probs, axis=1)
    uniform = np.full(7, 1.0 / 7)
    np.testing.assert_array_equal(injected_preds(probs, uniform), expected)


def test_large_prior_on_one_class_dominates() -> None:
    """A share profile overwhelmingly favouring class d turns every prediction into d."""
    rng = np.random.default_rng(1)
    probs = _random_probs(rng, 200, 7)
    shares = np.full(7, 1e-9)
    shares[3] = 1.0 - 6e-9
    preds = injected_preds(probs, shares)
    assert np.all(preds == 3)


def test_rho_shares_ratio_and_normalization() -> None:
    """rho_shares sums to 1 and its head/tail ratio equals rho."""
    shares = rho_shares(7, 100.0)
    assert shares.sum() == pytest.approx(1.0)
    assert shares[0] / shares[-1] == pytest.approx(100.0)
    assert np.all(np.diff(shares) <= 0)


def test_renormalize_subset_rows_sum_to_one() -> None:
    """Restricting to a column subset and renormalising always yields valid rows."""
    rng = np.random.default_rng(2)
    probs = _random_probs(rng, 50, 30)
    cols = rng.choice(30, size=7, replace=False)
    restricted = renormalize_subset(probs, cols)
    assert restricted.shape == (50, 7)
    np.testing.assert_allclose(restricted.sum(axis=1), 1.0, atol=1e-10)
