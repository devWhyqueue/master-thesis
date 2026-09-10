"""Unit tests for exp-4 class decodability components."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from decodability import K_VALUES, LAMBDAS, SUPPORTS
from decodability.audit import _verify_cell_audit
from decodability.evidence import CellEvidence
from decodability.linear import fit_multinomial_logistic
from decodability.neighbours import top_neighbours, vote
from decodability.probe import decode_shard_index
from decodability.select import _select_knn, _select_logreg


def test_top_neighbours_batched_vs_full():
    """Batched search matches brute-force full argsort ranking."""
    torch.manual_seed(42)
    np.random.seed(42)

    n_q, n_b, d = 20, 100, 16
    q = torch.randn(n_q, d)
    b = torch.randn(n_b, d)

    indices, sims, _ = top_neighbours(
        q, b, top=10, query_batch_size=5, bank_batch_size=25
    )

    q_n = q / torch.norm(q, dim=1, keepdim=True)
    b_n = b / torch.norm(b, dim=1, keepdim=True)
    full_sims = torch.matmul(q_n, b_n.T).numpy()

    for i in range(n_q):
        expected_order = np.lexsort((np.arange(n_b), -full_sims[i]))[:10]
        np.testing.assert_array_equal(indices[i], expected_order)
        np.testing.assert_allclose(sims[i], full_sims[i, expected_order], atol=1e-5)


def test_top_neighbours_duplicate_tie_order():
    """Duplicate bank vectors resolve ties by bank position (patch order)."""
    q = torch.tensor([[1.0, 0.0]])
    b = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

    indices, sims, _ = top_neighbours(
        q, b, top=2, query_batch_size=2, bank_batch_size=2
    )
    np.testing.assert_array_equal(indices[0], [0, 1])
    np.testing.assert_allclose(sims[0], [1.0, 1.0], atol=1e-5)


def test_vote_and_logreg_tie_breaking():
    """Ties in plurality voting break toward lower class index."""
    # Class 0 and Class 1 both have 2 votes
    labels = np.array([[0, 1, 0, 1]])
    preds, probs = vote(labels, k=4, n_classes=3)
    assert preds[0] == 0
    np.testing.assert_allclose(probs[0], [0.5, 0.5, 0.0])

    # Inverted order: still ties between 0 and 1
    labels2 = np.array([[1, 0, 1, 0]])
    preds2, _ = vote(labels2, k=4, n_classes=3)
    assert preds2[0] == 0


def test_logistic_c_and_regularization():
    """C == 1/(N*lambda) and norm of W decreases monotonically with lambda."""
    np.random.seed(42)
    n, d, c = 120, 10, 3
    x = np.random.randn(n, d)
    y = np.random.randint(0, c, size=n)

    lams = [1e-4, 1e-2, 1.0]
    weights = []
    for lam in lams:
        res = fit_multinomial_logistic(x, y, lambda_val=lam, max_iter=200)
        assert np.isclose(res.c_val, 1.0 / (n * lam))
        weights.append(np.linalg.norm(res.coef))

    assert weights[0] > weights[1] > weights[2]


def test_logistic_tolerance_is_on_mean_objective():
    """Solver gtol == tol * N, so the criterion does not tighten with sample count."""
    np.random.seed(7)
    x = np.random.randn(500, 8)
    y = np.random.randint(0, 3, size=500)

    res = fit_multinomial_logistic(x, y, lambda_val=1e-6, tol=1e-8, max_iter=300)
    assert np.isclose(res.solver_tolerance, 1e-8 * 500)
    assert res.converged


def test_selection_tie_rule_prefers_larger(tmp_path):
    """Differences <= 1e-10 resolve toward larger lambda / larger k."""
    from imbalance_benchmark.common import write_run_record

    config = {
        "paths": {"outputs": str(tmp_path)},
        "slurm": {"exp2_outputs": str(tmp_path)},
    }
    for s_idx in range(3):
        # 1e-6 and 1e-5 have identical validation accuracy
        for lam in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0):
            r_dir = (
                tmp_path
                / f"split={s_idx}"
                / "results"
                / "balanced"
                / "logreg"
                / f"lambda={lam}"
            )
            score = (
                0.80000000001
                if lam == 1e-6
                else (0.80000000000 if lam == 1e-5 else 0.5)
            )
            rec = {
                "solver": {"converged": True},
                "splits": {
                    "validation": {
                        "endpoints": {"patient_macro_balanced_accuracy": score}
                    }
                },
            }
            write_run_record(r_dir, rec)

        for k in K_VALUES:
            r_dir = (
                tmp_path / f"split={s_idx}" / "results" / "balanced" / "knn" / f"k={k}"
            )
            score = 0.75000000001 if k == 1 else (0.75000000000 if k == 5 else 0.4)
            rec = {
                "splits": {
                    "validation": {
                        "endpoints": {"patient_macro_balanced_accuracy": score}
                    }
                }
            }
            write_run_record(r_dir, rec)

    sel_l = _select_logreg(config, "balanced")
    assert sel_l["selected"] == 1e-5

    sel_k = _select_knn(config, "balanced")
    assert sel_k["selected"] == 5


def test_interaction_algebraic_identity():
    """Interaction identity I_h == B_mlp - B_h holds numerically."""
    # Synthetic replicate distributions
    np.random.seed(42)
    n_reps = 100
    a_mlp_c = np.random.uniform(0.6, 0.7, n_reps)
    a_mlp_s = np.random.uniform(0.7, 0.8, n_reps)
    a_h_c = np.random.uniform(0.62, 0.72, n_reps)
    a_h_s = np.random.uniform(0.71, 0.81, n_reps)

    g_c = a_h_c - a_mlp_c
    g_s = a_h_s - a_mlp_s
    b_h = a_h_s - a_h_c
    b_mlp = a_mlp_s - a_mlp_c
    i_h = g_c - g_s

    np.testing.assert_allclose(i_h, b_mlp - b_h, atol=1e-12)


def test_decode_shard_index_covers_grid_exactly_once():
    """All 48 probe-val shard indices cover 3 splits x 2 supports x 8 units once."""
    seen = set()
    for shard_index in range(48):
        split_index, support, unit = decode_shard_index(shard_index)
        assert split_index in range(3)
        assert support in SUPPORTS
        assert unit in range(len(LAMBDAS) + 1)
        seen.add((split_index, support, unit))

    assert len(seen) == 48
    assert seen == {
        (s, c, u) for s in range(3) for c in SUPPORTS for u in range(len(LAMBDAS) + 1)
    }


def test_preflight_detects_patient_overlap():
    """Preflight rejects CellEvidence if patients overlap between splits."""
    import pandas as pd
    import torch

    val_df = pd.DataFrame({"case_id": ["P1", "P2"], "slide_id": ["S1", "S2"]})
    test_df = pd.DataFrame({"case_id": ["P3", "P4"], "slide_id": ["S3", "S4"]})

    cell = CellEvidence(
        support="balanced",
        class_names=("A", "B"),
        train_x=torch.randn(100, 2560),
        train_y=np.zeros(100, dtype=int),
        train_patches=[f"pt_{i}" for i in range(100)],
        train_patients=["P1"] * 50 + ["P5"] * 50,  # P1 overlaps with validation
        val_x=torch.randn(10, 2560),
        val_y=np.zeros(10, dtype=int),
        val_identity=val_df,
        test_x=torch.randn(10, 2560),
        test_y=np.zeros(10, dtype=int),
        test_identity=test_df,
    )

    with pytest.raises(RuntimeError, match="Patient overlap detected"):
        _verify_cell_audit(cell)
