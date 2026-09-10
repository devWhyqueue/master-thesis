"""Unit tests for the patient-influence experiment's new components."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch
from decodability import allocation_manifest, exp2_split_paths
from decodability.evidence import CellEvidence
from decodability.linear import fit_multinomial_logistic
from imbalance_benchmark.common import compute_sha256, sign_file, write_run_record

from influence import MAX_ITER, SUPPORTS, TOLERANCE, exp3_root
from influence.baseline import verify_baseline
from influence.fit import FIT_SHARD_COUNT, decode_shard_index
from influence.weights import contribution_audit, patient_average_weights


def test_patient_average_weights_equal_contributions_all_ones():
    """Equal per-patient contributions within a class give unit weights."""
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    patients = ["p1", "p1", "p2", "p2", "p3", "p3", "p4", "p4"]
    weights = patient_average_weights(labels, patients)
    np.testing.assert_allclose(weights, np.ones(8))


def test_patient_average_weights_matches_hand_computation():
    """Skewed contributions match n_c / (G_c * m_ic) and preserve class/total mass."""
    labels = np.array([0, 0, 0, 0, 1, 1])
    patients = ["A", "A", "A", "B", "C", "C"]
    weights = patient_average_weights(labels, patients)
    np.testing.assert_allclose(weights, [2 / 3, 2 / 3, 2 / 3, 2.0, 1.0, 1.0])
    assert np.all(weights > 0)
    np.testing.assert_allclose(np.sum(weights[labels == 0]), 4.0)
    np.testing.assert_allclose(np.sum(weights[labels == 1]), 2.0)
    np.testing.assert_allclose(np.sum(weights), len(labels))


def test_contribution_audit_departure():
    """D_c is 0 under equal contributions and matches the hand-computed skewed value."""
    labels = np.array([0, 0, 0, 0, 1, 1])
    patients = ["A", "A", "A", "B", "C", "C"]
    rows = contribution_audit(labels, patients, ["a", "b"])
    by_class = {row["class"]: row for row in rows}
    assert by_class["a"]["G_c"] == 2
    assert by_class["a"]["n_c"] == 4
    assert np.isclose(by_class["a"]["D_c"], 0.25)
    assert by_class["b"]["G_c"] == 1
    assert np.isclose(by_class["b"]["D_c"], 0.0)


def test_uniform_sample_weight_reproduces_unweighted_fit():
    """All-ones sample_weight leaves the fitted coefficients unchanged."""
    np.random.seed(0)
    x = np.random.randn(80, 6)
    y = np.random.randint(0, 3, size=80)
    plain = fit_multinomial_logistic(x, y, lambda_val=1e-2, max_iter=500)
    weighted = fit_multinomial_logistic(
        x, y, lambda_val=1e-2, max_iter=500, sample_weight=np.ones(80)
    )
    np.testing.assert_allclose(plain.coef, weighted.coef, atol=1e-6)
    assert plain.weighted is False
    assert weighted.weighted is True


def test_decode_shard_index_covers_grid_exactly_once():
    """Fit shard indices cover 3 splits x 2 supports exactly once."""
    seen = set()
    for shard_index in range(FIT_SHARD_COUNT):
        split_index, support = decode_shard_index(shard_index)
        assert split_index in range(3)
        assert support in SUPPORTS
        seen.add((split_index, support))
    assert seen == {(s, c) for s in range(3) for c in SUPPORTS}


def test_contrast_identity_holds():
    """Interaction identity W_C - W_S == B_patch - B_patient holds numerically."""
    np.random.seed(1)
    n_reps = 50
    a_patch_c = np.random.uniform(0.6, 0.7, n_reps)
    a_patch_s = np.random.uniform(0.7, 0.8, n_reps)
    a_patient_c = np.random.uniform(0.62, 0.72, n_reps)
    a_patient_s = np.random.uniform(0.71, 0.81, n_reps)

    w_c = a_patient_c - a_patch_c
    w_s = a_patient_s - a_patch_s
    b_patch = a_patch_s - a_patch_c
    b_patient = a_patient_s - a_patient_c

    np.testing.assert_allclose(w_c - w_s, b_patch - b_patient, atol=1e-12)


def _build_baseline_fixture(tmp_path, support="balanced"):
    """Build a minimal tmp-path config, signed exp-3 selection, and preflight."""
    config = {
        "slurm": {
            "exp2_outputs": str(tmp_path / "exp2"),
            "exp3_outputs": str(tmp_path / "exp3"),
        },
        "feature_extraction": {"model_name": "virchow2"},
    }
    exp2_paths = exp2_split_paths(config, 0)
    manifest_path = allocation_manifest(exp2_paths, support)
    manifest_path.write_text("case_id,patch_id\nP1,x1\n", encoding="utf-8")

    lam, param_str = 0.1, "lambda=0.1"
    selection = {
        "supports": {support: {"logreg": {"selected": lam, "selected_param_str": param_str}}}
    }
    sel_path = exp3_root(config) / "data" / "probe_selection.json"
    sel_path.parent.mkdir(parents=True, exist_ok=True)
    sel_path.write_text(json.dumps(selection), encoding="utf-8")
    sign_file(sel_path)

    preflight = {
        "splits": {"0": {support: {"manifest_sha256": compute_sha256(manifest_path)}}}
    }
    preflight_path = exp3_root(config) / "data" / "preflight.json"
    preflight_path.write_text(json.dumps(preflight), encoding="utf-8")
    sign_file(preflight_path)

    return config, lam, param_str


def _write_baseline_record(
    config, support, param_str, n_train, test_y, lam, converged=True
):
    """Write a fake exp-3 baseline run.json for one support and param."""
    r_dir = exp3_root(config) / "split=0" / "results" / support / "logreg" / param_str
    record = {
        "feature_extraction": config["feature_extraction"],
        "solver": {
            "solver": "lbfgs",
            "precision": "float64",
            "tolerance": TOLERANCE,
            "solver_tolerance": TOLERANCE * n_train,
            "max_iter": MAX_ITER,
            "converged": converged,
            "lambda": lam,
            "C": 1.0 / (n_train * lam),
        },
        "splits": {
            "test": {"labels": test_y.tolist(), "preds": test_y.tolist()}
        },
    }
    write_run_record(r_dir, record)


def _build_cell(n_train, test_y, class_names=("a", "b")):
    """Build a minimal CellEvidence for baseline verification (only y/len fields used)."""
    empty_identity = pd.DataFrame({"case_id": [], "slide_id": [], "cancer_type": []})
    return CellEvidence(
        support="balanced",
        class_names=class_names,
        train_x=torch.zeros(n_train, 1),
        train_y=np.zeros(n_train, dtype=int),
        train_patches=[f"pt_{i}" for i in range(n_train)],
        train_patients=["P1"] * n_train,
        val_x=torch.zeros(1, 1),
        val_y=np.zeros(1, dtype=int),
        val_identity=empty_identity,
        test_x=torch.zeros(len(test_y), 1),
        test_y=test_y,
        test_identity=empty_identity,
    )


def test_verify_baseline_rejects_wrong_lambda(tmp_path):
    """A stored solver.lambda that differs from exp-3's selection is rejected."""
    config, lam, param_str = _build_baseline_fixture(tmp_path)
    test_y = np.array([0, 1, 0])
    _write_baseline_record(
        config, "balanced", param_str, n_train=10, test_y=test_y, lam=0.5
    )
    cell = _build_cell(n_train=10, test_y=test_y)
    with pytest.raises(RuntimeError, match="lambda"):
        verify_baseline(config, 0, "balanced", cell)


def test_verify_baseline_rejects_not_converged(tmp_path):
    """A baseline stored with converged: False is rejected."""
    config, lam, param_str = _build_baseline_fixture(tmp_path)
    test_y = np.array([0, 1, 0])
    _write_baseline_record(
        config,
        "balanced",
        param_str,
        n_train=10,
        test_y=test_y,
        lam=lam,
        converged=False,
    )
    cell = _build_cell(n_train=10, test_y=test_y)
    with pytest.raises(RuntimeError, match="converge"):
        verify_baseline(config, 0, "balanced", cell)
