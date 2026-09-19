"""Unit tests for directional centre shrinkage, its weight formula, fit selection, and gains."""

from __future__ import annotations

import numpy as np
import pytest

from imbalance_benchmark.common import ensure_dirs, read_run_record, split_paths, write_run_record

from sites import allocation_dir

from centre import PATIENT_COUNTS
from centre.cohort import TrainingTable
from centre.fit import split_arm

import directional.fit as fit_mod
from directional import ARMS, REUSED_ARM_FAMILIES
from directional.analyze import derived
from directional.centres import directional_weights, shrunk_centres
from directional.fit import _at_pending, _fit_at, select_alpha

_BASIS = np.array([[1.0, 0.0]])
_EIGVALS = np.array([0.5])
_MU_C = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 0.0]])


def _means(g: int) -> np.ndarray:
    """(C, g, d) patient means, every patient pinned to its class's toy centre."""
    return np.repeat(_MU_C[:, None, :], g, axis=1)


def test_shrunk_centres_identity_at_alpha_zero():
    """Alpha = 0 leaves centres exactly unchanged."""
    np.testing.assert_allclose(shrunk_centres(_means(2), _BASIS, _EIGVALS, 2, 0.0), _MU_C)


def test_shrunk_centres_unchanged_for_zero_rank_basis():
    """A rank-zero covariance basis returns the original centres."""
    empty_basis = np.empty((0, 2))
    empty_eigvals = np.empty(0)
    out = shrunk_centres(_means(2), empty_basis, empty_eigvals, 2, 1.0)
    np.testing.assert_allclose(out, _MU_C)


def test_shrunk_centres_leaves_nullspace_component_unchanged():
    """A direction outside ``basis`` (here the second axis) is never touched."""
    means = _means(2).copy()
    means[:, :, 1] = np.array([2.0, -3.0, 1.0])[:, None]
    out = shrunk_centres(means, _BASIS, _EIGVALS, 2, 1.0)
    np.testing.assert_allclose(out[:, 1], [2.0, -3.0, 1.0])


def test_shrunk_centres_preserves_grand_mean():
    """Every class shifts by weight * z_c along the same direction, so the grand mean is exact."""
    out = shrunk_centres(_means(2), _BASIS, _EIGVALS, 2, 1.0)
    np.testing.assert_allclose(out.mean(axis=0), _MU_C.mean(axis=0), atol=1e-12)


def test_directional_weights_and_shrunk_centres_match_hand_computation():
    """n_j = 0.25, signal = 1.0, s_j = 0.75 give weight 0.25, hand-verified against the shift."""
    weight = directional_weights(_means(2), _BASIS, _EIGVALS, 2)
    np.testing.assert_allclose(weight, [0.25])
    out = shrunk_centres(_means(2), _BASIS, _EIGVALS, 2, 1.0)
    np.testing.assert_allclose(out, [[0.75, 0.0], [-0.75, 0.0], [0.0, 0.0]])


def test_directional_weights_zero_when_denominator_is_zero():
    """No signal and no noise (zero eigenvalue, classes tied on that direction) gives weight zero."""
    means = np.repeat(np.zeros((3, 1, 2)), 2, axis=1)
    weight = directional_weights(means, _BASIS, np.array([0.0]), 2)
    np.testing.assert_allclose(weight, [0.0])


def test_select_alpha_ties_prefer_larger_alpha():
    """Equal validation scores among 0.0-2.0 with 4.0 trailing selects 2.0, not the smallest."""
    scores = {"0.0": 0.8, "0.5": 0.8, "1.0": 0.8, "2.0": 0.8, "4.0": 0.5}
    assert select_alpha(scores) == "2.0"


def test_arm_names_round_trip_through_split_arm():
    """ARMS and every reused-family arm decode to their own (family, G) via centre's split_arm."""
    reused = [
        f"{fam}{g}"
        for families in REUSED_ARM_FAMILIES.values()
        for fam in families
        for g in PATIENT_COUNTS
    ]
    for arm in (*ARMS, *reused):
        family, g = split_arm(arm)
        assert g in PATIENT_COUNTS
        assert arm == f"{family}{g}"
    assert split_arm("At10") == ("At", 10)


def test_derived_on_toy_arm_gaps_gives_hand_computed_gains_recovery_and_shares():
    """Gains, recovery fractions, and gap shares match hand-computed values on toy distributions."""
    base = {
        "R": (60.0, 66.0, 70.0),
        "C": (70.0, 73.0, 75.0),
        "S": (61.0, 66.5, 70.2),
        "St": (61.5, 66.8, 70.2),
        "RWc": (62.0, 67.0, 70.3),
        "A": (62.0, 67.0, 70.5),
        "At": (63.0, 67.5, 70.5),
    }
    arm = {
        f"{f}{g}": np.array([v])
        for f, vals in base.items()
        for g, v in zip(PATIENT_COUNTS, vals)
    }

    out = derived(arm)

    assert out["A5_minus_R5"][0] == pytest.approx(2.0)
    assert out["At5_minus_R5"][0] == pytest.approx(3.0)
    assert out["At5_minus_S5"][0] == pytest.approx(2.0)
    assert out["At5_minus_RWc5"][0] == pytest.approx(1.0)
    assert out["recovery_At_5"][0] == pytest.approx(0.3)
    assert out["recovery_At_10"][0] == pytest.approx(1.5 / 7.0)
    assert out["gap_A_5_to_10"][0] == pytest.approx(5.0)
    assert out["share_A_5_to_10"][0] == pytest.approx(1.0 / 6.0)


def _fake_run_record(score: float) -> dict:
    """Minimal well-formed run record with a given validation score, for baseline-reuse tests."""
    return {
        "dataset": {},
        "feature_extraction": {},
        "method": "logreg",
        "param": "lambda=1.0",
        "grid": {"g": 5, "m": 32, "draw": 0},
        "selected_lambda": 1.0,
        "solver": {},
        "arm": "R5",
        "splits": {
            "validation": {"endpoints": {"patient_macro_balanced_accuracy": score}},
            "test": {
                "endpoints": {},
                "labels": [0, 1],
                "preds": [0, 1],
                "probabilities": [[1.0, 0.0], [0.0, 1.0]],
            },
        },
    }


def test_at_pending_tracks_missing_nonzero_alpha_factors(tmp_path):
    """An At record is pending until every nonzero alpha factor has a validation score."""
    out_dir = tmp_path / "at5"
    assert _at_pending(out_dir) is True

    write_run_record(
        out_dir,
        {
            **_fake_run_record(0.6),
            "arm": "At5",
            "alpha_factor": 0.5,
            "alpha_validation_scores": {"0.0": 0.5, "0.5": 0.6, "1.0": 0.55, "2.0": 0.5},
        },
    )
    assert _at_pending(out_dir) is True

    write_run_record(
        out_dir,
        {
            **_fake_run_record(0.6),
            "arm": "At5",
            "alpha_factor": 0.5,
            "alpha_validation_scores": {
                "0.0": 0.5,
                "0.5": 0.6,
                "1.0": 0.55,
                "2.0": 0.5,
                "4.0": 0.4,
            },
        },
    )
    assert _at_pending(out_dir) is False


def test_fit_at_reuses_baseline_r_record_when_alpha_zero_wins(tmp_path, monkeypatch):
    """When every nonzero alpha scores below R's, At's record is R's, copied verbatim."""
    exp16_out = tmp_path / "exp16"
    exp20_out = tmp_path / "exp20"
    config = {
        "paths": {"outputs": str(exp20_out)},
        "slurm": {"baseline_outputs": str(exp16_out)},
    }
    baseline_paths = split_paths(ensure_dirs({"paths": {"outputs": str(exp16_out)}}), 0)
    write_run_record(allocation_dir(baseline_paths, "R5", 0), _fake_run_record(0.9))

    monkeypatch.setattr(fit_mod, "shrunk_centres", lambda *a, **k: np.zeros((1, 1)))
    monkeypatch.setattr(fit_mod, "move_centres", lambda x, y, target: target)
    monkeypatch.setattr(
        fit_mod,
        "tune_and_fit_draw",
        lambda x, y, evals: (
            "fit",
            1.0,
            "preds",
            "probs",
            {"patient_macro_balanced_accuracy": 0.1},
            {},
        ),
    )

    table = TrainingTable(np.zeros((2, 2)), np.array([0, 1]))
    out_dir = allocation_dir(split_paths(ensure_dirs(config), 0), "At5", 0)
    _fit_at(
        config, out_dir, "At5", table, means=None, basis=None, eigvals=None,
        evals=None, draw_idx=0, g=5, split_idx=0,
    )

    record = read_run_record(out_dir, array_fields=())
    assert record["arm"] == "At5"
    assert record["alpha_factor"] == pytest.approx(0.0)
    assert record["alpha_validation_scores"]["0.0"] == pytest.approx(0.9)
