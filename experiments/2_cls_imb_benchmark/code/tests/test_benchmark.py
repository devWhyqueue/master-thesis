from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from imbalance_benchmark.construction import (
    split_cases,
    allocate_counts,
    select_patches_round_robin,
)
from imbalance_benchmark.modeling.models import (
    MLP,
)
from imbalance_benchmark.modeling.losses import (
    FocalLoss,
    ScholzCombinedLoss,
)
from imbalance_benchmark.analysis import (
    expected_calibration_error,
    brier_score,
    holm_adjust_pvalues,
)
from imbalance_benchmark.analysis.inference.permutation import (
    paired_block_permutation_ba,
)


def test_split_cases():
    # Generate dummy patient/slide mapping
    rows = []
    for pat_idx in range(20):
        pat_id = f"PAT_{pat_idx}"
        cls = "class_A" if pat_idx < 10 else "class_B"
        rows.append(
            {"case_id": pat_id, "slide_id": f"SLIDE_{pat_id}", "cancer_type": cls}
        )
    df = pd.DataFrame(rows)

    # Split
    df_splits = split_cases(df, seed=42)

    # Check split assignment
    assert "split" in df_splits.columns
    assert set(df_splits["split"].unique()).issubset({"train", "validation", "test"})

    # Patient disjoint validation
    grouped = df_splits.groupby("case_id")["split"].nunique()
    assert (grouped == 1).all()


def test_allocate_counts():
    # 4 classes with these available counts
    available = [100, 80, 50, 40]
    total_T = 160

    # rho = 1.0 (balanced)
    counts = allocate_counts(available, total_T, rho=1.0, min_support=10)
    assert sum(counts) == total_T
    assert all(c >= 10 for c in counts)
    assert all(c <= a for c, a in zip(counts, available))

    # rho = 10.0 (moderate)
    counts = allocate_counts(available, total_T, rho=10.0, min_support=10)
    assert sum(counts) == total_T
    assert all(c >= 10 for c in counts)
    assert all(c <= a for c, a in zip(counts, available))
    assert counts[0] > counts[-1]


def test_round_robin():
    rows = []
    for i in range(100):
        rows.append(
            {
                "case_id": f"PAT_{i // 10}",
                "slide_id": f"SLIDE_{i // 5}",
                "patch_id": f"PATCH_{i}",
                "cancer_type": "class_A",
            }
        )
    df = pd.DataFrame(rows)

    # Select 20 patches round robin
    selected = select_patches_round_robin(df, 20, seed=42)
    assert len(selected) == 20

    # Patient cap: max patient patches <= max(1, 20 * 0.10) = 2 patches
    pat_counts = selected["case_id"].value_counts()
    assert (pat_counts <= 2).all()


def test_models_and_losses():
    # Test MLP forward
    model = MLP(input_dim=10, hidden_dim=8, output_dim=3, dropout=0.0)
    x = torch.randn(5, 10)
    out = model(x)
    assert out.shape == (5, 3)

    # Test FocalLoss
    criterion = FocalLoss(gamma=2.0)
    targets = torch.tensor([0, 1, 2, 1, 0])
    loss = criterion(out, targets)
    assert loss.item() >= 0.0

    # Test ScholzCombinedLoss (F1)
    combined_criterion = ScholzCombinedLoss(n_classes=3, metric="f1", weight=1.0)
    loss = combined_criterion(out, targets)
    assert loss.item() >= 0.0


def test_calibration_and_permutation():
    # ECE and Brier score
    probs = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.1, 0.8]])
    targets = np.array([0, 1, 2])
    ece = expected_calibration_error(targets, probs, n_bins=5)
    assert 0.0 <= ece <= 1.0

    brier = brier_score(targets, probs, n_classes=3)
    assert brier >= 0.0

    # Paired patient-block permutation test
    labels = np.array([0, 1, 0, 1])
    case_ids = np.array(["P0", "P1", "P2", "P3"])
    method_preds = np.array([0, 1, 0, 1])
    ce_preds = np.array([1, 0, 1, 0])
    p_val = paired_block_permutation_ba(
        labels, method_preds, ce_preds, case_ids, n_classes=2, n_permutations=100
    )
    assert 0.0 <= p_val <= 1.0

    # Holm adjustment
    p_vals = [0.01, 0.04, 0.02, 0.15]
    adj = holm_adjust_pvalues(p_vals)
    assert len(adj) == 4
    assert all(a >= p for a, p in zip(adj, p_vals))
