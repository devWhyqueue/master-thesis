"""Unit tests for coverage-similarity geometry, search, census, and precision logic."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from coverage_redundancy.quantities import omega_similarity

from similarity import MIN_R_GAP
from similarity.census import _draw_checks, _manipulation_gaps
from similarity.geometry import ClassGeometry, omega_of, r_of
from similarity.candidates import _farthest_first
from similarity.search import CohortResult, _corner_loss_fn, _swap_search
from similarity.simulate import _max_t_critical
from similarity.weights import _class_stratified_weights


def _geometry(
    pool: list[str], d_pool: np.ndarray, gram: np.ndarray, tau2: float = 1.0
) -> ClassGeometry:
    return ClassGeometry(pool=pool, d_pool=d_pool, d_val=d_pool[:1], gram=gram, tau2=tau2, rho=0.1)


def test_gram_omega_matches_omega_similarity():
    """Gram omega equals coverage_redundancy.quantities.omega_similarity."""
    rng = np.random.default_rng(0)
    pool = [f"p{i}" for i in range(8)]
    vecs = rng.normal(size=(8, 6))
    means = {(p, "cls"): vecs[i] for i, p in enumerate(pool)}
    tau2 = 1.0
    idx = np.array([0, 2, 3, 5])
    dev = vecs - vecs.mean(axis=0)
    geo = _geometry(pool, np.zeros((8, 8)), dev @ dev.T, tau2)

    expected = omega_similarity(means, [pool[i] for i in idx], pool, "cls", tau2)
    assert omega_of(geo, idx) == pytest.approx(expected)


def test_split_geometry_omega_uses_raw_means(monkeypatch):
    """build_split_geometry's Gram uses raw means, matching tau2's units, not unit embeddings."""
    from types import SimpleNamespace

    import similarity.geometry as geometry

    rng = np.random.default_rng(1)
    pool = [f"p{i}" for i in range(8)]
    raw = 50.0 * rng.normal(size=(8, 6))
    unit = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    means = {(p, "cls"): raw[i] for i, p in enumerate(pool)}
    ctx = SimpleNamespace(
        class_names=["cls"],
        eligible_by_class={"cls": pool},
        embeddings={(p, "cls"): unit[i] for i, p in enumerate(pool)},
        means=means,
        tau2_by_class={"cls": 400.0},
        rho_by_class={"cls": 0.1},
    )
    monkeypatch.setattr(geometry, "_build_split_context", lambda *args: ctx)
    monkeypatch.setattr(geometry, "_patients_of", lambda df, part, c: pool[:2])

    geo = geometry.build_split_geometry({}, 0, None, None, ())["cls"]
    idx = np.array([1, 4, 6])
    expected = omega_similarity(means, [pool[i] for i in idx], pool, "cls", 400.0)
    assert omega_of(geo, idx) == pytest.approx(expected)


def test_farthest_first_adds_farthest_patient():
    """Farthest-first adds the patient farthest from the selected set."""
    pool = ["a", "b", "c", "d"]
    d_pool = np.array(
        [[0, 1, 2, 10], [1, 0, 1, 9], [2, 1, 0, 8], [10, 9, 8, 0]], dtype=float
    )
    geo = _geometry(pool, d_pool, np.zeros((4, 4)))
    result = _farthest_first(0, geo, 2)
    assert result[0] == 0
    assert result[1] == 3  # "d" is farthest from "a"


def test_swap_search_reaches_reachable_target():
    """Swap search reaches a reachable (r, omega) target on a toy geometry."""
    positions = np.array([0.0, 1.0, 2.0, 10.0, 20.0, 30.0])
    pool = [f"p{i}" for i in range(6)]
    d_pool = np.abs(positions[:, None] - positions[None, :])
    dev = positions - positions.mean()
    geo = _geometry(pool, d_pool, np.outer(dev, dev))

    target_idx = np.array([0, 1, 2])
    loss_fn = _corner_loss_fn(r_of(geo.d_pool, target_idx), omega_of(geo, target_idx))

    # One swap away from the target: exercises the swap mechanism on a genuinely
    # reachable case, rather than a landscape needing several uphill swaps first.
    result = _swap_search(np.array([0, 1, 3]), geo, loss_fn)
    assert set(result.tolist()) == {0, 1, 2}


def _cohort(r_train: float, r_val: float, omega: float, neff: float) -> CohortResult:
    return CohortResult(patients=["x"], r_train=r_train, r_val=r_val, omega=omega, neff=neff)


def _passing_results() -> dict[str, CohortResult]:
    return {
        "good_low": _cohort(0.30, 0.30, 0.10, 20.0),
        "poor_low": _cohort(0.34, 0.34, 0.11, 18.0),
        "good_high": _cohort(0.30, 0.30, 0.18, 15.0),
        "poor_high": _cohort(0.34, 0.34, 0.185, 14.0),
        "ten_match": _cohort(0.30, 0.30, 0.12, 19.9),
    }


def test_manipulation_check_pass():
    """The manipulation-check truth table: a matched, well-separated cohort set passes."""
    assert all(_draw_checks(_passing_results()).values())


def test_manipulation_check_tolerance_fail():
    """The manipulation-check truth table: an out-of-tolerance coverage pair fails."""
    results = _passing_results()
    results["good_high"] = _cohort(0.30, 0.31, 0.18, 15.0)  # r_val gap 0.01 > R_TOL
    checks = _draw_checks(results)
    assert checks["good_low_vs_good_high_r_val"] is False


def test_manipulation_check_gap_fail():
    """The manipulation-check truth table: an insufficiently manipulated gap fails."""
    results = {
        "good_low": _cohort(0.30, 0.30, 0.10, 20.0),
        "poor_low": _cohort(0.302, 0.302, 0.10, 20.0),  # gap 0.002 < MIN_R_GAP
        "good_high": _cohort(0.30, 0.30, 0.10, 20.0),
        "poor_high": _cohort(0.302, 0.302, 0.10, 20.0),
    }
    gaps = _manipulation_gaps(results)
    assert gaps["coverage_poor_minus_good_low"] < MIN_R_GAP


def test_stratified_weights_column0_ones_and_class_sums():
    """Stratified weights: column 0 is ones, and per-class column sums equal n_c."""
    identity = pd.DataFrame(
        {
            "case_id": ["a", "a", "b", "b", "c", "d", "d"],
            "cancer_type": ["t1", "t1", "t1", "t1", "t1", "t2", "t2"],
        }
    )
    position, weights = _class_stratified_weights(identity, n_replicates=50, seed=0)
    assert (weights[:, 0] == 1.0).all()

    t1_idx = np.array([position[c] for c in ("a", "b", "c")])
    t2_idx = np.array([position[c] for c in ("d",)])
    assert np.allclose(weights[t1_idx, 1:].sum(axis=0), 3.0)
    assert np.allclose(weights[t2_idx, 1:].sum(axis=0), 1.0)


def test_max_t_critical_value_single_contrast():
    """Max-t critical value is approximately 1.96 for one contrast on normal replicates."""
    rng = np.random.default_rng(0)
    reps = rng.normal(size=20000)
    base = {"only": np.concatenate([[0.0], reps])}
    assert _max_t_critical(base) == pytest.approx(1.96, abs=0.05)
