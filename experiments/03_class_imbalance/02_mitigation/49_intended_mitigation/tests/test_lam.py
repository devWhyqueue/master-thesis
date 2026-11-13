"""Per-lambda logistic path, exp-25 selection rule, and the fixed-offset fit of gate 4."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest
from decodability.linear import fit_multinomial_logistic, predict_logreg

from breadth import LAMBDAS
from breadth.fit import _macro_recall_mean, _select_best_lambda

from intended.lam import fit_lambda_path, fit_offset_logistic, logits, select_lambda

K, DIM = 3, 5


def _data(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Overlapping Gaussian classes with unequal counts; ``cases`` groups rows into patients."""
    rng = np.random.default_rng(seed)
    y = rng.choice(K, size=n, p=[0.6, 0.3, 0.1])
    centres = np.random.default_rng(99).normal(scale=1.5, size=(K, DIM))
    x = centres[y] + rng.normal(size=(n, DIM))
    cases = np.array([f"c{y_i}_{i % 4}" for i, y_i in enumerate(y)])
    return x, y, cases


@pytest.fixture(scope="module")
def world() -> dict:
    x, y, _ = _data(300, 0)
    vx, vy, vcases = _data(120, 1)
    return {"x": x, "y": y, "vx": vx, "vy": vy, "vcases": vcases, "fits": fit_lambda_path(x, y)}


def _score(fit, world: dict) -> float:
    preds, _ = predict_logreg(world["vx"], fit.coef, fit.intercept)
    return _macro_recall_mean(world["vy"], preds, world["vcases"])


def test_select_matches_exp25(world: dict) -> None:
    identity = pd.DataFrame({"case_id": world["vcases"], "slide_id": world["vcases"]})
    best_fit, best_lam, _ = _select_best_lambda(world["x"], world["y"], world["vx"], world["vy"], identity)
    index = select_lambda(world["fits"], world["vx"], world["vy"], world["vcases"])
    assert LAMBDAS[index] == best_lam
    ours = world["fits"][index]
    np.testing.assert_array_equal(
        predict_logreg(world["vx"], ours.coef, ours.intercept)[0],
        predict_logreg(world["vx"], best_fit.coef, best_fit.intercept)[0],
    )
    np.testing.assert_array_equal(logits(ours, world["vx"]).argmax(1), predict_logreg(world["vx"], ours.coef, ours.intercept)[0])


def test_ties_prefer_larger_lambda(world: dict) -> None:
    good = world["fits"][4]
    bad = dataclasses.replace(good, coef=np.zeros_like(good.coef), intercept=np.array([5.0, 0.0, 0.0]))
    assert _score(bad, world) < _score(good, world)
    fits = [bad] * len(LAMBDAS)
    fits[2] = fits[5] = good
    assert select_lambda(fits, world["vx"], world["vy"], world["vcases"]) == 5


def test_nonconverged_skipped(world: dict) -> None:
    fits = list(world["fits"])
    best = select_lambda(fits, world["vx"], world["vy"], world["vcases"])
    fits[best] = dataclasses.replace(fits[best], converged=False)
    chosen = select_lambda(fits, world["vx"], world["vy"], world["vcases"])
    scores = [_score(f, world) if f.converged else -1.0 for f in fits]
    expected = max(i for i, s in enumerate(scores) if s >= max(scores) - 1e-10)
    assert chosen == expected and chosen != best
    with pytest.raises(RuntimeError, match="converged"):
        select_lambda([dataclasses.replace(f, converged=False) for f in fits], world["vx"], world["vy"], world["vcases"])


def test_offset_shifts_selection(world: dict) -> None:
    fits = world["fits"]
    offset = np.array([-1e6, 0.0, 1e6])  # forces class 2 everywhere: every lambda scores 1/K
    assert select_lambda(fits, world["vx"], world["vy"], world["vcases"], offset=offset) == len(LAMBDAS) - 1


def test_offset_fit_equals_posthoc(world: dict) -> None:
    tau, lam = 1.0, 1e-2
    x, y = world["x"], world["y"]
    log_pi = np.log(np.bincount(y, minlength=K) / len(y))
    plain = world["fits"][list(LAMBDAS).index(lam)]
    # exp-25's tolerance stops near gradient 3e-7; the identity itself is checked at 1e-12.
    exact = fit_multinomial_logistic(x, y, lam, tol=1e-12)
    fit = fit_offset_logistic(x, y, lam, tau * log_pi)
    coef, intercept = fit.coef, fit.intercept
    assert fit.converged and fit.gradient_max < 1e-8
    np.testing.assert_allclose(coef, exact.coef, atol=1e-5)
    shifted = exact.intercept - tau * log_pi
    # Softmax is shift-invariant: only intercept differences between classes are identified.
    np.testing.assert_allclose(intercept - intercept.mean(), shifted - shifted.mean(), atol=1e-5)
    posthoc = logits(plain, world["vx"]) - tau * log_pi
    trained = world["vx"] @ coef.T + intercept
    top2 = np.sort(posthoc, axis=1)[:, -2:]
    clear = (top2[:, 1] - top2[:, 0]) > 1e-6
    np.testing.assert_array_equal(posthoc.argmax(1)[clear], trained.argmax(1)[clear])


def test_offset_fit_warm_restarts_past_iteration_cap(world: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("intended.lam.MAX_ITER", 5)
    x, y = world["x"], world["y"]
    fit = fit_offset_logistic(x, y, 1e-2, np.zeros(K))
    assert fit.iterations > 5  # a single capped run cannot exceed 5

