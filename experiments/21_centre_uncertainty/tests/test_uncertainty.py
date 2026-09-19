"""Unit tests for the centre-uncertainty loss, fit selection, freezing, and contrasts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from decodability.linear import fit_multinomial_logistic, predict_logreg
from scipy.optimize import approx_fprime

from breadth.fit import EvalPartition
from centre import PATIENT_COUNTS
from centre.cohort import TrainingTable
from centre.fit import split_arm

import uncertainty.fit as fit_mod
from uncertainty import ARMS, NONZERO_T_FACTORS, REUSED_ARM_FAMILIES
from uncertainty.analyze import derived
from uncertainty.fit import _extend_selection, _fit_grid
from uncertainty.freeze import (
    Candidate,
    fingerprint,
    read_selection,
    select_candidates,
    t_grid,
    write_selection,
)
from uncertainty.loss import (
    Covariance,
    UncertainFit,
    covariance_for,
    fit_uncertain_logistic,
    loss_and_grad,
    pair_quadratics,
)

K, D, N = 3, 6, 40
GRID = (0.0, 1.0)


def _data(seed: int = 0):
    rng = np.random.default_rng(seed)
    y = np.arange(N) % K
    x = rng.normal(size=(N, D)) + 0.8 * np.eye(K, D)[y]
    basis = np.linalg.qr(rng.normal(size=(D, 3)))[0].T
    return x, y, basis, np.array([0.9, 0.4, 0.1])


def _dense(basis, variances, iso=0.0):
    return basis.T @ np.diag(variances) @ basis + iso * np.eye(basis.shape[1])


def test_analytic_gradient_matches_finite_differences():
    """Both covariance kinds: analytic gradient equals central differences."""
    x, y, basis, eig = _data()
    theta = np.random.default_rng(1).normal(size=K * D + K) * 0.3
    for cov in (
        covariance_for("patient", basis, eig, 5),
        covariance_for("isotropic", basis, eig, 5),
        Covariance(basis, eig / 5, 0.05),
    ):
        args = (x, y, K, 0.1, 2.0, cov)
        _, grad = loss_and_grad(theta, *args)
        num = approx_fprime(theta, lambda th: loss_and_grad(th, *args)[0], 1e-6)
        np.testing.assert_allclose(grad, num, atol=1e-5)


def test_zero_strength_matches_ordinary_logistic_regression():
    """t = 0 and a zero covariance both reproduce ``fit_multinomial_logistic`` predictions."""
    x, y, basis, eig = _data()
    ref = fit_multinomial_logistic(x, y, lambda_val=0.05, tol=1e-10)
    ref_pred, _ = predict_logreg(x, ref.coef, ref.intercept)
    zero = Covariance(basis, np.zeros(3), 0.0)
    assert zero.is_zero
    for t, cov in ((0.0, covariance_for("patient", basis, eig, 5)), (4.0, zero)):
        fit = fit_uncertain_logistic(x, y, K, 0.05, t, cov, (1e-10, 10000))
        assert fit.converged
        np.testing.assert_allclose(fit.coef, ref.coef, atol=1e-5)
        pred, _ = predict_logreg(x, fit.coef, fit.intercept)
        assert (pred == ref_pred).all()


def test_low_rank_quadratics_match_dense_and_isotropic_preserves_trace():
    """Basis-evaluated pair quadratics equal a dense V; the control keeps tr(V)."""
    _, _, basis, eig = _data()
    w = np.random.default_rng(2).normal(size=(K, D))
    patient = covariance_for("patient", basis, eig, 5)
    dense = _dense(basis, eig / 5)
    expected = np.array(
        [[(w[k] - w[c]) @ dense @ (w[k] - w[c]) for c in range(K)] for k in range(K)]
    )
    np.testing.assert_allclose(pair_quadratics(w, patient), expected, atol=1e-12)
    iso = covariance_for("isotropic", basis, eig, 5)
    assert iso.isotropic * D == pytest.approx(np.trace(dense))
    d = w[0] - w[1]
    assert pair_quadratics(w, iso)[0, 1] == pytest.approx(iso.isotropic * d @ d)


def test_rank_deficient_and_zero_covariance_are_explicit():
    """No eigen-directions gives a zero covariance that fits like ordinary logistic regression."""
    x, y, basis, _ = _data()
    empty = covariance_for("patient", basis[:0], np.empty(0), 5)
    assert empty.is_zero
    assert pair_quadratics(np.ones((K, D)), empty).shape == (K, K)
    fit = fit_uncertain_logistic(x, y, K, 0.1, 1.0, empty, (1e-8, 1000))
    assert fit.converged


def test_invalid_inputs_raise_and_nonconvergence_is_reported():
    """Bad lambda, t, labels, features, or covariance raise; a tiny iteration cap flags nonconvergence."""
    x, y, basis, eig = _data()
    cov = covariance_for("patient", basis, eig, 5)
    for kwargs in (
        {"lam": 0.0},
        {"lam": -1.0},
        {"t": -1.0},
        {"y": y + K},
        {"x": np.where(np.eye(N, D) > 0, np.nan, x)},
    ):
        args = {"x": x, "y": y, "lam": 0.1, "t": 1.0, **kwargs}
        with pytest.raises(ValueError):
            fit_uncertain_logistic(
                args["x"], args["y"], K, args["lam"], args["t"], cov, (1e-8, 100)
            )
    with pytest.raises(ValueError):
        covariance_for("patient", basis, -eig, 5)
    with pytest.raises(ValueError):
        covariance_for("other", basis, eig, 5)
    assert not fit_uncertain_logistic(x, y, K, 1e-6, 1.0, cov, (1e-12, 2)).converged


def _cand(kind, t, lam, score, converged=True):
    return Candidate(kind, t, lam, score, {"converged": converged})


def test_selection_ties_prefer_smaller_t_then_larger_lambda_and_baseline():
    """Equal scores pick the smallest t, then largest lambda; the t = 0 baseline wins ties; nonconverged skipped."""
    cands = [
        _cand("patient", 0.25, 0.1, 0.7),
        _cand("patient", 0.25, 10.0, 0.7),
        _cand("patient", 1.0, 1.0, 0.7),
        _cand("patient", 4.0, 1.0, 0.9, converged=False),
    ]
    best = select_candidates(cands, None)
    assert (best.t, best.lam) == (0.25, 10.0)
    assert select_candidates(cands, 0.7) is None
    assert select_candidates(cands, 0.6) is best
    with pytest.raises(RuntimeError):
        select_candidates([_cand("patient", 1.0, 1.0, 0.5, converged=False)], None)


def _evals(seed=0, shift=0.0):
    rng = np.random.default_rng(seed)
    y = np.arange(12) % 3
    ident = pd.DataFrame(
        {"case_id": [f"p{i}" for i in range(12)], "slide_id": [f"s{i}" for i in range(12)]}
    )
    x = rng.normal(size=(12, 8)) + shift
    return EvalPartition(x, y, ident, x, y, ident)


def test_covariance_geometry_uses_training_features_only(monkeypatch):
    """Changing validation features changes scores but never the covariance geometry."""
    monkeypatch.setattr(fit_mod, "LAMBDAS", (0.1, 1.0))
    rng = np.random.default_rng(3)
    g, m = 2, 3
    y = np.repeat(np.arange(3), g * m)
    table = TrainingTable(rng.normal(size=(len(y), 8)) + y[:, None] * 0.5, y)
    _, fits_a, geo_a = _fit_grid(table, 3, g, _evals(0))
    cand_b, fits_b, geo_b = _fit_grid(table, 3, g, _evals(0, shift=5.0))
    assert geo_a == geo_b
    assert len(cand_b) == 2 * len(NONZERO_T_FACTORS) * 2
    key = ("patient", 1.0, 0.1)
    np.testing.assert_allclose(fits_a[key].coef, fits_b[key].coef)


def test_frozen_selection_round_trips_and_rejects_stale_evidence(tmp_path):
    """A matching fingerprint reads back; a changed cohort or a missing winners file is rejected."""
    table = TrainingTable(np.ones((4, 2)), np.array([0, 0, 1, 1]))
    fp = fingerprint({}, table, 5, {"r": 0.5}, GRID)
    assert fp != fingerprint({}, TrainingTable(table.x + 1, table.y), 5, {"r": 0.5}, GRID)
    assert fp != fingerprint({}, table, 5, {"r": 0.6}, GRID)
    assert read_selection(tmp_path, lambda _: fp) is None
    fit = UncertainFit(np.zeros((2, 2)), np.zeros(2), 0.0, 0.0, 1, True)
    cand = _cand("patient", 1.0, 0.1, 0.8)
    selected = {"U": {"kind": "patient", "t": 1.0, "lam": 0.1}, "Ut": None, "It": None}
    write_selection(tmp_path, fp, [cand], selected, {}, 0.5, {("patient", 1.0, 0.1): fit})
    assert read_selection(tmp_path, lambda _: fp)["selected"] == selected
    with pytest.raises(RuntimeError):
        read_selection(tmp_path, lambda _: "other")
    (tmp_path / "winners.npz").unlink()
    with pytest.raises(RuntimeError):
        read_selection(tmp_path, lambda _: fp)


def test_arm_names_round_trip_through_split_arm():
    """ARMS and every reused arm decode to their own (family, G)."""
    reused = [
        f"{fam}{g}"
        for families in REUSED_ARM_FAMILIES.values()
        for fam in families
        for g in PATIENT_COUNTS
    ]
    for arm in (*ARMS, *reused):
        family, g = split_arm(arm)
        assert arm == f"{family}{g}" and g in PATIENT_COUNTS


def test_derived_contrasts_and_gap_reduction_on_toy_distributions():
    """Gains, contrasts, recovery, and paired gap reduction match hand-computed values."""
    base = {
        "R": (60.0, 66.0, 70.0),
        "C": (70.0, 73.0, 75.0),
        "S": (61.0, 66.5, 70.2),
        "St": (61.5, 66.8, 70.2),
        "A": (62.0, 67.0, 70.5),
        "At": (63.0, 67.5, 70.5),
        "RWc": (62.0, 67.0, 70.3),
        "U": (61.0, 66.5, 70.1),
        "Ut": (64.0, 68.0, 70.5),
        "It": (62.0, 66.5, 70.1),
    }
    arm = {
        f"{f}{g}": np.array([v])
        for f, vals in base.items()
        for g, v in zip(PATIENT_COUNTS, vals)
    }
    out = derived(arm)
    assert out["Ut5_minus_R5"][0] == pytest.approx(4.0)
    assert out["Ut5_minus_It5"][0] == pytest.approx(2.0)
    assert out["Ut5_minus_RWc5"][0] == pytest.approx(2.0)
    assert out["Ut5_minus_At5"][0] == pytest.approx(1.0)
    assert out["recovery_Ut_5"][0] == pytest.approx(0.4)
    assert out["gap_Ut_5_to_20"][0] == pytest.approx(6.5)
    assert out["gap_reduction_Ut_5_to_20"][0] == pytest.approx(3.5)
    assert out["share_Ut_5_to_20"][0] == pytest.approx(0.35)


def test_extension_fits_only_new_strengths_and_reselects_over_the_merged_grid(
    tmp_path, monkeypatch
):
    """A frozen t = 1 winner is kept unless a newly fitted t = 16 candidate beats it strictly."""
    fit = UncertainFit(np.zeros((2, 2)), np.zeros(2), 0.0, 0.0, 1, True)
    old_cand = _cand("patient", 1.0, 0.1, 0.8)
    sel = {"U": {"kind": "patient", "t": 1.0, "lam": 0.1}, "Ut": None, "It": None}
    sel["Ut"] = sel["U"]
    key = ("patient", 1.0, 0.1)
    write_selection(tmp_path, "fp", [old_cand], sel, {}, 0.5, {key: fit})
    old = read_selection(tmp_path, lambda _: "fp")
    assert t_grid(old) == (0.0, 1.0)
    calls = []
    for score, expected_t in ((0.9, 16.0), (0.8, 1.0)):
        new_key = ("patient", 16.0, 0.1)
        new = _cand("patient", 16.0, 0.1, score)
        monkeypatch.setattr(
            fit_mod,
            "_fit_grid",
            lambda *a, _n=new, _k=new_key: calls.append(a[-1]) or ([_n], {_k: fit}, {}),
        )
        chosen = _extend_selection(None, 3, None, old, (16.0,), (tmp_path, "fp2", 5, 0.5))
        assert chosen["Ut"]["t"] == expected_t and chosen["U"]["t"] == 1.0
    assert calls == [(16.0,), (16.0,)]
