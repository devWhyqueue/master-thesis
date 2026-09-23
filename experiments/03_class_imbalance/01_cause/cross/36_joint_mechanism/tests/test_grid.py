"""Unit tests for the experiment-local lambda grid: selection, fixed-lambda reuse, and the
write/read roundtrip that backs resumable shards and gate 5's boundary-stability check.
"""

from __future__ import annotations

import numpy as np
import pytest
from decodability.linear import LinearFitResult

from joint.grid import LAMBDAS, GridCandidate, read_grid, select_at_lambda, select_best, write_grid


def _candidate(lam: float, val_score: float, converged: bool = True, d: int = 3, c: int = 4):
    fit = LinearFitResult(
        coef=np.full((c, d), lam),
        intercept=np.full(c, lam),
        lambda_val=lam,
        c_val=1.0 / lam,
        solver="lbfgs",
        precision="float64",
        tolerance=1e-8,
        solver_tolerance=1e-8,
        max_iter=10000,
        n_iter=5,
        objective=0.1,
        converged=converged,
        weighted=False,
    )
    return GridCandidate(lam, fit, val_score)


def test_lambdas_span_1e_minus_8_to_1e2():
    """PLAN.md line 54: powers of ten from 1e-8 through 1e2."""
    assert LAMBDAS[0] == pytest.approx(1e-8)
    assert LAMBDAS[-1] == pytest.approx(1e2)
    assert len(LAMBDAS) == 11


def test_select_best_ignores_non_converged_candidates():
    candidates = [_candidate(1e-8, 0.9, converged=False), _candidate(1e-7, 0.5)]
    assert select_best(candidates).lambda_val == 1e-7


def test_select_best_breaks_ties_toward_larger_lambda():
    candidates = [_candidate(1e-6, 0.7), _candidate(1e-5, 0.7)]
    assert select_best(candidates).lambda_val == 1e-5


def test_select_at_lambda_returns_the_exact_candidate_without_refitting():
    """The fixed-regularization control (PLAN.md line 39) reuses an already-fit candidate."""
    candidates = [_candidate(lam, float(i)) for i, lam in enumerate(LAMBDAS)]
    picked = select_at_lambda(candidates, LAMBDAS[3])
    assert picked.lambda_val == LAMBDAS[3]
    assert picked.val_score == 3.0


def test_select_at_lambda_raises_for_an_unknown_lambda():
    candidates = [_candidate(1e-6, 0.1)]
    with pytest.raises(ValueError):
        select_at_lambda(candidates, 5.0)


def test_write_read_grid_roundtrip(tmp_path):
    """Every candidate's coefficients and metadata survive a write/read roundtrip exactly."""
    candidates = [_candidate(lam, float(i)) for i, lam in enumerate(LAMBDAS[:3])]
    write_grid(tmp_path, candidates)
    loaded = read_grid(tmp_path)
    assert len(loaded) == len(candidates)
    for original, back in zip(candidates, loaded):
        assert back.lambda_val == original.lambda_val
        assert back.val_score == original.val_score
        assert back.fit.converged == original.fit.converged
        np.testing.assert_array_equal(back.fit.coef, original.fit.coef)
        np.testing.assert_array_equal(back.fit.intercept, original.fit.intercept)
