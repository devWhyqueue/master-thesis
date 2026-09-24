"""Unit tests for the cross-fit and naive lambda-selection oracles, and the phase-1 label rule."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from decodability.linear import LinearFitResult

from breadth.fit import EvalPartition

from joint.grid import GridCandidate

from selection_rule.analyze import _label
from selection_rule.oracle import crossfit_predictions, fold_split, naive_oracle


def _candidate(lam: float, coef: np.ndarray, intercept: np.ndarray) -> GridCandidate:
    fit = LinearFitResult(
        coef=coef,
        intercept=intercept,
        lambda_val=lam,
        c_val=1.0 / lam,
        solver="lbfgs",
        precision="float64",
        tolerance=1e-8,
        solver_tolerance=1e-8,
        max_iter=10000,
        n_iter=5,
        objective=0.1,
        converged=True,
        weighted=False,
    )
    return GridCandidate(lam, fit, val_score=0.0)


def _evals(x: np.ndarray, y: np.ndarray, case_ids: list[str]) -> EvalPartition:
    ids = pd.DataFrame({"case_id": case_ids, "slide_id": case_ids})
    return EvalPartition(x, y, ids, x, y, ids)


# Two-class, 1-feature toy candidates: "pos" predicts class 1 iff x > 0, "neg" the opposite.
_POS = _candidate(1.0, coef=np.array([[0.0], [10.0]]), intercept=np.zeros(2))
_NEG = _candidate(2.0, coef=np.array([[10.0], [0.0]]), intercept=np.zeros(2))
_MID = _candidate(3.0, coef=np.array([[0.0], [-100.0]]), intercept=np.zeros(2))


def test_fold_split_is_disjoint_covers_every_patient_and_every_class():
    case_ids = [f"p{i}" for i in range(20)]
    y = np.array([i % 3 for i in range(20)])
    ids = pd.DataFrame({"case_id": case_ids})
    mask_a, mask_b = fold_split(ids, y, seed=0)

    assert not np.any(mask_a & mask_b)
    assert np.all(mask_a | mask_b)
    for cls in np.unique(y):
        assert mask_a[y == cls].any()
        assert mask_b[y == cls].any()


def test_crossfit_never_predicts_a_fold_with_its_own_winner(monkeypatch):
    """Toy grid where the best lambda differs per fold (PLAN.md "Refinements" #1)."""
    # Fold A: x=+1, y=1 -- matches "pos"'s rule, "neg" gets it wrong.
    # Fold B: x=-1, y=1 -- matches "neg"'s rule, "pos" gets it wrong.
    x = np.array([[1.0], [1.0], [-1.0], [-1.0]])
    y = np.array([1, 1, 1, 1])
    evals = _evals(x, y, ["p0", "p1", "p2", "p3"])
    mask_a = np.array([True, True, False, False])
    mask_b = ~mask_a
    monkeypatch.setattr(
        "selection_rule.oracle.fold_split", lambda *_a, **_k: (mask_a, mask_b)
    )

    test_preds, _, winner_a, winner_b = crossfit_predictions([_POS, _NEG], evals, seed=0)

    assert winner_a.lambda_val == _POS.lambda_val  # "pos" wins fold A on its own rows
    assert winner_b.lambda_val == _NEG.lambda_val  # "neg" wins fold B on its own rows
    # Fold A is predicted by "neg" (the out-of-fold winner), not by "pos" (its own winner):
    # "neg" predicts class 0 at x=+1, so fold A's stitched predictions come out wrong.
    assert np.all(test_preds[mask_a] == 0)
    # Fold B is predicted by "pos" (out-of-fold), which predicts class 0 at x=-1.
    assert np.all(test_preds[mask_b] == 0)


def test_naive_oracle_achieves_the_maximum_full_test_score():
    x = np.array([[1.0], [1.0], [-1.0], [-1.0]])
    y = np.array([1, 1, 1, 1])
    evals = _evals(x, y, ["p0", "p1", "p2", "p3"])
    candidates = [_POS, _NEG, _MID]

    winner = naive_oracle(candidates, evals)

    from decodability.linear import predict_logreg
    from imbalance_benchmark.analysis.reporting.clustered_endpoints import (
        _macro_recall_mean,
    )

    cases = evals.test_id["case_id"].astype(str).to_numpy()
    scores = {}
    for cand in candidates:
        preds, _ = predict_logreg(evals.test_x, cand.fit.coef, cand.fit.intercept)
        scores[cand.lambda_val] = _macro_recall_mean(evals.test_y, preds, cases)
    assert scores[winner.lambda_val] == max(scores.values())
    assert scores[winner.lambda_val] >= scores[_MID.lambda_val]  # >= an arbitrary "tuned" pick


@pytest.mark.parametrize(
    ("gap_ci", "delta_ci", "closure", "expected"),
    [
        ((-1.0, 1.0), (0.5, 2.0), 0.6, "no_prior_gap"),
        ((1.0, 2.0), (-0.5, 0.5), 0.6, "prior_gap_intrinsic"),
        ((1.0, 2.0), (0.5, 2.0), 0.6, "selection_explains"),
        ((1.0, 2.0), (0.1, 0.4), 0.2, "selection_contributes"),
    ],
)
def test_label_covers_every_precondition_case(gap_ci, delta_ci, closure, expected):
    gap_tuned = {"point": (gap_ci[0] + gap_ci[1]) / 2, "ci_lower": gap_ci[0], "ci_upper": gap_ci[1]}
    delta = {"point": (delta_ci[0] + delta_ci[1]) / 2, "ci_lower": delta_ci[0], "ci_upper": delta_ci[1]}
    assert _label(gap_tuned, delta, closure) == expected
