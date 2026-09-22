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
from imbalance_benchmark.analysis.reporting.tables import rq3_table
from imbalance_benchmark.modeling.workflows.run_context import (
    RunExposure,
    cost_payload,
)
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
    context.case_codes, _ = pd.factorize(context.case_ids, sort=False)
    context.slide_codes, _ = pd.factorize(context.slide_ids, sort=False)
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
    context.case_codes, _ = pd.factorize(context.case_ids, sort=False)
    context.slide_codes, _ = pd.factorize(context.slide_ids, sort=False)
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
    context.case_codes, _ = pd.factorize(context.case_ids, sort=False)
    context.slide_codes, _ = pd.factorize(context.slide_ids, sort=False)
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


def test_rq3_table_reports_two_four_signal_models() -> None:
    fit = {
        "intercept": 0.0,
        "slopes": [1.0, 2.0, 3.0, 4.0],
        "sigma_u": 0.2,
        "sigma": 0.1,
    }

    table = rq3_table({"damage": fit, "recovery": fit})

    assert "damage" in table and "recovery" in table
    assert "slope\\_independent\\_shortage" in table
    assert "slope\\_support\\_difficulty\\_alignment" in table
    assert "slope\\_diversity\\_shortage" in table
    assert "gate\\_pass" not in table

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


def test_class_sums_matches_looped_sums_per_code() -> None:
    rng = np.random.default_rng(0)
    n_rows, n_patients, n_replicates, n_codes = 40, 12, 5, 4
    row_patient = rng.integers(0, n_patients, size=n_rows)
    patient = rng.random((n_patients, n_replicates))
    weights = PatientWeights(row_patient, patient)
    codes = rng.integers(0, n_codes, size=n_rows)
    values = rng.random(n_rows)

    got = weights.class_sums(values, codes, n_codes)
    expected = np.stack([weights.sums(values, mask=codes == c) for c in range(n_codes)])
    assert np.allclose(got, expected, atol=1e-12)

    got_ones = weights.class_sums(1.0, codes, n_codes)
    expected_ones = np.stack([weights.sums(1.0, mask=codes == c) for c in range(n_codes)])
    assert np.allclose(got_ones, expected_ones, atol=1e-12)


def test_class_metrics_matches_naive_per_class_masks() -> None:
    """Pins `_class_metrics`/`_probability_class_metrics` against the mask-per-class
    `weighted_mean` loop they replaced (plan "Speed up analyze-combine" §2)."""
    from imbalance_benchmark.analysis.reporting.secondary_intervals.metrics import (
        _class_metrics,
    )
    from imbalance_benchmark.analysis.reporting.secondary_intervals.probability import (
        _probability_class_metrics,
    )
    from imbalance_benchmark.analysis.reporting.secondary_intervals.weighted import (
        weighted_mean,
    )

    rng = np.random.default_rng(5)
    n_rows, n_patients, n_replicates, n_classes = 30, 10, 4, 3
    row_patient = rng.integers(0, n_patients, size=n_rows)
    patient = rng.random((n_patients, n_replicates))
    weights = PatientWeights(row_patient, patient)
    labels = rng.integers(0, n_classes, size=n_rows)
    predictions = rng.integers(0, n_classes, size=n_rows)
    probabilities = rng.dirichlet(np.ones(n_classes), size=n_rows)
    class_names = [f"c{i}" for i in range(n_classes)]

    got_prob = _probability_class_metrics(labels, probabilities, weights, class_names)
    got_class = _class_metrics(labels, predictions, probabilities, weights, class_names)

    true_probability = np.clip(probabilities[np.arange(n_rows), labels], 1e-12, 1.0)
    nll_values = -np.log(true_probability)
    one_hot = np.eye(n_classes)[labels]
    brier_values = np.sum((probabilities - one_hot) ** 2, axis=1)
    correct = (predictions == labels).astype(float)

    nll_by_class, f1_by_class, recall_by_class = [], [], []
    for class_index, class_name in enumerate(class_names):
        true_class = labels == class_index
        predicted_class = predictions == class_index
        class_nll = weighted_mean(nll_values, weights, true_class)
        class_brier = weighted_mean(brier_values, weights, true_class)
        assert np.allclose(got_prob[f"nll:{class_name}"], class_nll, equal_nan=True)
        assert np.allclose(got_prob[f"brier:{class_name}"], class_brier, equal_nan=True)

        recall = weighted_mean(correct, weights, true_class)
        precision = weighted_mean(true_class.astype(float), weights, predicted_class)
        with np.errstate(divide="ignore", invalid="ignore"):
            f1 = np.where(
                precision + recall > 0,
                2 * precision * recall / (precision + recall),
                0.0,
            )
        assert np.allclose(got_class[f"recall:{class_name}"], recall, equal_nan=True)
        assert np.allclose(got_class[f"f1:{class_name}"], f1, equal_nan=True)
        nll_by_class.append(class_nll)
        f1_by_class.append(f1)
        recall_by_class.append(recall)

    assert np.allclose(
        got_prob["macro_nll"], np.nanmean(np.stack(nll_by_class), axis=0), equal_nan=True
    )
    assert np.allclose(
        got_class["macro_f1"], np.nanmean(np.stack(f1_by_class), axis=0), equal_nan=True
    )
    assert np.allclose(
        got_class["balanced_accuracy"],
        np.nanmean(np.stack(recall_by_class), axis=0),
        equal_nan=True,
    )


def test_group_balanced_accuracy_matches_naive_per_class_group_mean() -> None:
    """Pins `_group_balanced_accuracy` against the per-class `pd.factorize` + masked
    `_group_mean` loop it replaced (plan §2)."""
    from imbalance_benchmark.analysis.reporting.secondary_intervals.metrics import (
        _group_balanced_accuracy,
    )

    rng = np.random.default_rng(6)
    n_rows, n_patients, n_replicates, n_classes = 24, 8, 3, 3
    row_patient = rng.integers(0, n_patients, size=n_rows)
    patient = rng.random((n_patients, n_replicates))
    weights = PatientWeights(row_patient, patient)
    labels = rng.integers(0, n_classes, size=n_rows)
    predictions = rng.integers(0, n_classes, size=n_rows)
    groups = rng.integers(0, 5, size=n_rows).astype(str)
    codes, _ = pd.factorize(groups, sort=False)
    n_groups = int(codes.max()) + 1

    got = _group_balanced_accuracy(
        labels, predictions, weights, codes, n_groups, n_classes
    )

    correct = (predictions == labels).astype(float)
    recalls = []
    for class_index in np.unique(labels):
        selected = labels == class_index
        local_codes, _ = pd.factorize(groups[selected], sort=False)
        counts = np.bincount(local_codes)
        scale = np.zeros(n_rows, dtype=np.float64)
        scale[selected] = 1.0 / counts[local_codes]
        numerator = weights.sums(correct * scale)
        denominator = weights.sums(scale)
        with np.errstate(divide="ignore", invalid="ignore"):
            recalls.append(np.where(denominator > 0, numerator / denominator, np.nan))
    expected = np.nanmean(np.stack(recalls), axis=0)
    assert np.allclose(got, expected, equal_nan=True)


def test_group_macro_f1_matches_sklearn_per_group() -> None:
    """Covers the case a class is absent from a group's truth but present in its
    predictions -- the corner `_group_macro_f1`'s `2*tp/(support+predicted_count)`
    identity must still match sklearn's `zero_division=0` fallback on (plan §4)."""
    from sklearn.metrics import f1_score

    from imbalance_benchmark.analysis.reporting.secondary_intervals.metrics import (
        _group_macro_f1,
    )

    n_classes = 3
    codes = np.array([0, 0, 0, 1, 1, 1, 1])
    labels = np.array([0, 0, 1, 2, 2, 2, 2])
    # group 1: class 0 is absent from truth but predicted once; class 1 absent
    # from both truth and predictions (must be excluded from the average).
    predictions = np.array([0, 1, 1, 0, 2, 2, 2])
    weights = PatientWeights(np.arange(len(codes)), np.ones((len(codes), 2)))

    got = _group_macro_f1(
        labels, predictions, weights, codes, n_groups=2, n_classes=n_classes
    )

    expected_per_group = [
        f1_score(
            labels[codes == group],
            predictions[codes == group],
            average="macro",
            zero_division=0,  # type: ignore
        )
        for group in range(2)
    ]
    expected = float(np.mean(expected_per_group))
    assert got[0] == pytest.approx(expected)
    assert got[1] == pytest.approx(expected)
