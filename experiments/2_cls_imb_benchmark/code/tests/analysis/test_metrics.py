from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from imbalance_benchmark.analysis.inference.bootstrap import PatientWeights
from imbalance_benchmark.analysis.inference.context import BootstrapContext
from imbalance_benchmark.analysis.reporting.clustered_endpoints import (
    clustered_endpoints,
)
from imbalance_benchmark.modeling.context import RunExposure, cost_payload
from imbalance_benchmark.modeling.models import AttentionMil
from imbalance_benchmark.modeling.training import _fit_step

def _unit_weights(n_rows: int, n_replicates: int) -> PatientWeights:
    """One patient per row (unit weight everywhere), matching the old all-ones matrix."""
    return PatientWeights(
        np.arange(n_rows), np.ones((n_rows, n_replicates), dtype=np.float64)
    )

def _bootstrap_context() -> BootstrapContext:
    context = object.__new__(BootstrapContext)
    context.case_ids = np.array(["c1", "c1", "c2", "c2"])
    context.slide_ids = np.array(["s1", "s1", "s2", "s2"])
    context.weights = _unit_weights(4, 3)
    context.n_replicates = 3
    context._seed = 7
    context._seed_indices = {}
    return context

def _mil_context(method: str) -> dict[str, object]:
    return {
        "is_mil": True,
        "method": method,
        "device": torch.device("cpu"),
        "model": AttentionMil(2, 3, 2, dropout=0.0),
        "criterion": torch.nn.CrossEntropyLoss(),
        "param": 0.1,
        "method_diagnostics": {},
    }

def test_secondary_bootstrap_includes_cluster_macro_endpoints() -> None:
    labels = np.array([0, 1, 1, 0])
    predictions = np.array([[0, 0, 1, 1]])
    probabilities = np.array(
        [[[0.9, 0.1], [0.6, 0.4], [0.2, 0.8], [0.3, 0.7]]]
    )
    identity = pd.DataFrame(
        {
            "case_id": ["c1", "c1", "c1", "c2"],
            "slide_id": ["s1", "s1", "s2", "s3"],
        }
    )
    context = object.__new__(BootstrapContext)
    context.case_ids = identity["case_id"].to_numpy()
    context.slide_ids = identity["slide_id"].to_numpy()
    context.weights = _unit_weights(4, 3)
    context.n_replicates = 3
    context._seed = 7
    context._seed_indices = {}

    result = context.secondary_distributions(
        labels, predictions, probabilities, ["A", "B"], {}
    )
    observed = clustered_endpoints(
        labels, predictions[0], probabilities[0], identity
    )

    for endpoint in (
        "patch_micro_accuracy",
        "slide_macro_accuracy",
        "patient_macro_accuracy",
        "slide_macro_balanced_accuracy",
        "patient_macro_balanced_accuracy",
        "slide_macro_f1",
        "patient_macro_f1",
        "slide_macro_nll",
        "patient_macro_nll",
        "slide_macro_brier",
        "patient_macro_brier",
    ):
        assert result[endpoint][0] == pytest.approx(observed[endpoint])

def test_wsi_secondary_outputs_use_only_applicable_endpoint_names() -> None:
    labels = np.array([0, 1])
    predictions = np.array([[0, 1]])
    probabilities = np.array([[[0.9, 0.1], [0.1, 0.9]]])
    context = object.__new__(BootstrapContext)
    context.case_ids = np.array(["c1", "c2"])
    context.slide_ids = np.array(["s1", "s2"])
    context.weights = _unit_weights(2, 2)
    context.n_replicates = 2
    context._seed = 7
    context._seed_indices = {}

    tcga = context.secondary_distributions(
        labels,
        predictions,
        probabilities,
        ["LUAD", "LUSC"],
        {},
        is_mil=True,
        ordinal=False,
    )
    panda = context.secondary_distributions(
        labels,
        predictions,
        probabilities,
        ["ISUP0", "ISUP1"],
        {},
        is_mil=True,
        ordinal=True,
    )

    assert "patch_micro_accuracy" not in tcga
    assert "quadratic_weighted_kappa" not in tcga
    assert "ordinal_mean_absolute_error" not in tcga
    assert "patch_micro_accuracy" not in panda
    assert "quadratic_weighted_kappa" in panda
    assert "ordinal_mean_absolute_error" in panda

def test_wsi_run_endpoints_do_not_call_slide_accuracy_patch_accuracy() -> None:
    identity = pd.DataFrame(
        {"case_id": ["c1", "c2"], "slide_id": ["s1", "s2"]}
    )

    endpoints = clustered_endpoints(
        np.array([0, 1]),
        np.array([0, 1]),
        np.array([[0.9, 0.1], [0.1, 0.9]]),
        identity,
        is_mil=True,
    )

    assert "patch_micro_accuracy" not in endpoints

def test_cluster_macro_balanced_accuracy_is_macro_recall_after_cluster_aggregation() -> (
    None
):
    labels = np.array([0, 0, 1])
    predictions = np.array([0, 0, 0])
    probabilities = np.eye(2)[predictions]
    identity = pd.DataFrame(
        {"case_id": ["p0", "p1", "p2"], "slide_id": ["s0", "s1", "s2"]}
    )

    endpoints = clustered_endpoints(labels, predictions, probabilities, identity)

    assert endpoints["slide_macro_balanced_accuracy"] == pytest.approx(0.5)

def test_secondary_endpoint_distributions_cover_classwise_and_calibration() -> None:
    labels = np.array([0, 0, 1, 1])
    predictions = np.array([labels, labels])
    probabilities = np.stack([np.eye(2)[labels], np.eye(2)[labels]])

    distributions = _bootstrap_context().secondary_distributions(
        labels,
        predictions,
        probabilities,
        ["head", "tail"],
        {"head": "head", "tail": "tail"},
    )

    assert {
        "macro_f1",
        "negative_log_likelihood",
        "macro_nll",
        "brier_score",
        "expected_calibration_error",
        "recall:head",
        "f1:tail",
        "nll:tail",
        "brier:head",
        "tier_recall:tail",
        "tier_nll:tail",
        "tier_brier:tail",
    } <= set(distributions)
    assert all(len(values) == 3 for values in distributions.values())

def test_secondary_endpoint_rows_retain_effect_estimates_and_intervals() -> None:
    from imbalance_benchmark.analysis.reporting.secondary_intervals.report import (
        _endpoint_row,
    )

    row = _endpoint_row(
        ("native", "severe", "focal"),
        "macro_f1",
        np.array([0.8, 0.7, 0.9]),
        ("native", "severe", "ce"),
        {"macro_f1": np.array([0.6, 0.5, 0.7])},
    )

    assert row["reference"] == "native/severe/ce"
    assert row["effect"] == pytest.approx(0.2)
    assert row["effect_ci_low"] == pytest.approx(0.2)
    assert row["effect_ci_high"] == pytest.approx(0.2)

def test_cost_comparisons_report_paired_effect_intervals() -> None:
    from imbalance_benchmark.analysis.reporting.secondary_intervals.costs import (
        cost_comparison_rows,
    )

    rows = []
    for split in range(3):
        for seed, ce_cost, focal_cost in ((0, 3.0, 4.0), (1, 5.0, 8.0)):
            rows.extend(
                [
                    {
                        "patient_split": split,
                        "assignment": "native",
                        "condition": "severe",
                        "method": "ce",
                        "seed_index": seed,
                        "wall_clock_seconds": ce_cost,
                    },
                    {
                        "patient_split": split,
                        "assignment": "native",
                        "condition": "severe",
                        "method": "focal",
                        "seed_index": seed,
                        "wall_clock_seconds": focal_cost,
                    },
                ]
            )

    result = cost_comparison_rows(pd.DataFrame(rows), 100, seed=3)

    assert len(result) == 1
    assert result[0]["effect"] == pytest.approx(2.0)
    assert result[0]["reference"] == "severe/ce"
    assert result[0]["ci_low"] <= result[0]["effect"] <= result[0]["ci_high"]

def test_mil_exposure_counts_processed_instances_not_only_bags() -> None:
    context = _mil_context("ce")

    _fit_step(
        ([torch.ones(2, 2), torch.ones(3, 2)], torch.tensor([0, 1])),
        context,
        step=0,
        max_steps=1,
    )

    assert context["processed_examples"] == 2
    assert context["processed_instances"] == 5

def test_clustered_endpoints_report_slide_and_patient_macro_nll_and_brier() -> None:
    from imbalance_benchmark.analysis.reporting.clustered_endpoints import (
        clustered_endpoints,
    )

    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([[0.8, 0.2], [0.6, 0.4], [0.3, 0.7], [0.1, 0.9]])
    predictions = probabilities.argmax(axis=1)
    identity = pd.DataFrame(
        {
            "case_id": ["p0", "p0", "p1", "p1"],
            "slide_id": ["s0", "s1", "s2", "s3"],
        }
    )

    out = clustered_endpoints(labels, predictions, probabilities, identity, seed=0)

    for key in (
        "slide_macro_balanced_accuracy",
        "patient_macro_balanced_accuracy",
        "slide_macro_f1",
        "patient_macro_f1",
        "slide_macro_nll",
        "patient_macro_nll",
        "slide_macro_brier",
        "patient_macro_brier",
    ):
        assert key in out and np.isfinite(out[key])

def test_weighted_ece_matches_scalar_ece_at_unit_weights() -> None:
    """The crossed-bootstrap ECE reduces to the scalar fixed-bin ECE at unit weights."""
    from imbalance_benchmark.analysis.inference.bootstrap import weighted_ece
    from imbalance_benchmark.analysis.metrics import expected_calibration_error

    rng = np.random.default_rng(0)
    labels = rng.integers(0, 3, size=50)
    probs = rng.dirichlet(np.ones(3), size=50)
    weights = _unit_weights(50, 1)

    weighted = weighted_ece(labels, probs, weights)[0]

    assert weighted == pytest.approx(expected_calibration_error(labels, probs))

def test_tail_recall_is_grouped_by_assignment(tmp_path: Path) -> None:
    """Tail recall must not be averaged across tail assignments and copied per row."""
    import sqlite3

    from imbalance_benchmark.analysis.db import init_schema
    from imbalance_benchmark.analysis.reporting.tables import _with_tail_recall

    conn = sqlite3.connect(":memory:")
    init_schema(conn)
    # Same class 'C' is tail under assignment 'a' with recall 0.2 and under 'b'
    # with recall 0.8; grouping without assignment would report 0.5 for both.
    conn.executemany(
        "INSERT INTO runs (run_id, result_dir, benchmark, condition, assignment, method, seed_index) "
        "VALUES (?,?,?,?,?,?,?)",
        [
            ("run_a", "d", "patch", "severe", "a", "weighted_ce", 0),
            ("run_b", "d", "patch", "severe", "b", "weighted_ce", 0),
        ],
    )
    conn.executemany(
        "INSERT INTO eval_classwise (run_id, split, class_name, tier, precision, recall, f1, support, nll, brier) "
        "VALUES (?,?,?,?,?,?,?,?,?,?)",
        [
            ("run_a", "test", "C", "tail", 0.0, 0.2, 0.0, 5, 0.0, 0.0),
            ("run_b", "test", "C", "tail", 0.0, 0.8, 0.0, 5, 0.0, 0.0),
        ],
    )
    conn.commit()

    summary = pd.DataFrame(
        {
            "assignment": ["a", "b"],
            "condition": ["severe", "severe"],
            "method": ["weighted_ce", "weighted_ce"],
        }
    )
    merged = _with_tail_recall(summary, conn, "test").set_index("assignment")

    assert merged.loc["a", "tail_recall"] == pytest.approx(0.2)
    assert merged.loc["b", "tail_recall"] == pytest.approx(0.8)

def test_mil_covariates_exclude_patch_effective_support(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Effective support is a patch-only sensitivity covariate."""
    from imbalance_benchmark.analysis.predictors.rq3_analysis import _covariates

    manifest = tmp_path / "manifest.csv"
    balanced = tmp_path / "manifest_balanced.csv"
    condition = tmp_path / "condition.csv"
    rows = [
        {"case_id": "p0", "slide_id": "s0", "cancer_type": "A", "feature_path": "a.pt"},
        {"case_id": "p0", "slide_id": "s0", "cancer_type": "A", "feature_path": "b.pt"},
        {"case_id": "p1", "slide_id": "s1", "cancer_type": "B", "feature_path": "c.pt"},
    ]
    pd.DataFrame(rows).to_csv(manifest, index=False)
    pd.DataFrame(rows).to_csv(balanced, index=False)
    pd.DataFrame(rows).to_csv(condition, index=False)

    features = np.array([[1.0, 0.0], [0.0, 1.0]])
    labels = np.array([0, 1])
    monkeypatch.setattr(
        "imbalance_benchmark.analysis.predictors.rq3_analysis._feature_frame",
        lambda *_: (features, labels),
    )
    monkeypatch.setattr(
        "imbalance_benchmark.analysis.predictors.rq3_analysis.intrinsic_separability",
        lambda *_: {
            "linear_probe_macro_recall": 0.5,
            "knn_macro_recall": 0.5,
            "per_class_nn_error": {},
        },
    )
    monkeypatch.setattr(
        "imbalance_benchmark.analysis.predictors.rq3_analysis.condition_learnability",
        lambda *_: {"linear_probe_macro_recall": 0.5},
    )
    monkeypatch.setattr(
        "imbalance_benchmark.analysis.predictors.rq3_analysis.class_margin_cross_fit",
        lambda *_: np.array([0.1, 0.2]),
    )

    result = _covariates(
        {"data": tmp_path}, True, {"path": str(condition), "contribution_stats": {}}
    )

    assert "log_effective_support" not in result

def test_cost_uses_actual_processed_examples_for_partial_batches() -> None:
    cost = cost_payload(
        "ce",
        budget=3,
        elapsed=1.0,
        model=torch.nn.Linear(2, 2),
        exposure=RunExposure(
            unique_examples=10, exposed_examples=10, processed_examples=10
        ),
    )

    assert cost["processed_examples"] == 10
    assert cost["effective_passes_through_unique_examples"] == 1.0

def test_post_hoc_cost_does_not_inherit_a_previous_gpu_memory_peak(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 123)

    cost = cost_payload(
        "post_hoc_logit_adjustment",
        budget=0,
        elapsed=0.0,
        model=torch.nn.Linear(2, 2),
        exposure=RunExposure(
            unique_examples=4,
            exposed_examples=0,
            processed_examples=0,
            peak_memory_bytes=0,
        ),
    )

    assert cost["peak_accelerator_memory_bytes"] == 0
