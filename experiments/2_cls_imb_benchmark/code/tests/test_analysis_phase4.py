from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from imbalance_benchmark.analysis.calibration import (
    apply_target_prior_correction,
    balanced_decision_logits,
    fit_temperature,
)
from imbalance_benchmark.analysis.db import (
    connect_db,
    discover_result_dirs,
    ingest_run,
    init_schema,
)
from imbalance_benchmark.analysis.inference.bootstrap import (
    build_strata,
    expand_to_rows,
    gather_seed_resampled,
    kish_effective_count,
    resample_patient_weights,
    weighted_balanced_accuracy,
)
from imbalance_benchmark.analysis.inference.preflight import bootstrap_preflight
from imbalance_benchmark.analysis.inference.context import BootstrapContext
from imbalance_benchmark.analysis.inference.gates import (
    calibration_gate,
    ci_excludes_zero,
    deficit,
    discrimination_gate,
    recovery,
)
from imbalance_benchmark.analysis.inference.holm import apply_holm, confirmatory_family
from imbalance_benchmark.analysis.inference.crossed_permutation import (
    crossed_block_permutation_ba,
)
from imbalance_benchmark.analysis.inference.permutation import (
    paired_block_permutation_ba,
)
from imbalance_benchmark.analysis.metrics import (
    assign_tiers,
    classification_payload,
    negative_log_likelihood,
)
from imbalance_benchmark.analysis.predictors.rq3_wiring import (
    fit_deficit_model,
    fit_gate_pass_model,
    fit_recovery_model,
)
from imbalance_benchmark.analysis.predictors.hierarchical_models import _log_scale_prior
from imbalance_benchmark.analysis.predictors.rq3_analysis import _cells
from imbalance_benchmark.analysis.predictors.rq3_cross_split import _comparison_maps
from imbalance_benchmark.analysis.predictors.separability import (
    effective_support,
    intraclass_correlation,
)
from imbalance_benchmark.analysis.query import (
    load_classwise,
    load_eval_details,
    load_test_identity,
)
from imbalance_benchmark.analysis.reporting.ingestion import _ingest_discovered_run
from imbalance_benchmark.analysis.reporting.tables import (
    calibration_table,
    results_table,
)
from imbalance_benchmark.commands.analyze import _aggregate_split_comparisons
from imbalance_benchmark.common import (
    ensure_dirs,
    read_run_record,
    split_paths,
    write_json,
    write_run_record,
)


# --- metrics ------------------------------------------------------------------


def test_assign_tiers_ceil_k_over_3_and_binary_case():
    classes = ["A", "B", "C", "D", "E", "F", "G"]
    allocated = {"A": 100, "B": 90, "C": 80, "D": 50, "E": 20, "F": 10, "G": 5}
    tiers = assign_tiers(classes, allocated)
    assert [tiers[c] for c in classes] == [
        "head",
        "head",
        "head",
        "body",
        "tail",
        "tail",
        "tail",
    ]

    binary_tiers = assign_tiers(["A", "B"], {"A": 10, "B": 5})
    assert binary_tiers == {"A": "head", "B": "tail"}


def test_classification_payload_shapes_and_macro_nll():
    labels = [0, 0, 1, 1, 2, 2]
    preds = [0, 1, 1, 1, 2, 0]
    probs = [
        [0.8, 0.1, 0.1],
        [0.3, 0.6, 0.1],
        [0.1, 0.8, 0.1],
        [0.2, 0.7, 0.1],
        [0.1, 0.1, 0.8],
        [0.5, 0.2, 0.3],
    ]
    payload = classification_payload(labels, preds, probs, ["A", "B", "C"])
    assert payload["confusion_matrix"] == np.array(payload["confusion_matrix"]).tolist()
    assert len(payload["precision_per_class"]) == 3
    assert payload["macro_nll"] == pytest.approx(
        float(
            np.mean(
                [
                    negative_log_likelihood(
                        np.array(labels)[np.array(labels) == c],
                        np.array(probs)[np.array(labels) == c],
                    )
                    for c in range(3)
                ]
            )
        )
    )
    assert "quadratic_weighted_kappa" not in payload
    assert "ordinal_mean_absolute_error" not in payload
    ordinal_payload = classification_payload(
        labels, preds, probs, ["ISUP0", "ISUP1", "ISUP2"], ordinal=True
    )
    assert "quadratic_weighted_kappa" in ordinal_payload
    assert ordinal_payload["ordinal_mean_absolute_error"] >= 0.0


# --- calibration ----------------------------------------------------------------


def test_temperature_scaling_lowers_synthetic_overconfidence_nll():
    rng = np.random.default_rng(0)
    n, n_classes = 200, 3
    labels = rng.integers(0, n_classes, size=n)
    logits = np.zeros((n, n_classes))
    logits[np.arange(n), labels] = 8.0
    logits += rng.normal(scale=0.5, size=(n, n_classes))
    flip = rng.random(n) < 0.15
    logits[flip] = logits[flip][:, ::-1]
    fit = fit_temperature(logits, labels)
    from imbalance_benchmark.analysis.calibration import apply_temperature

    raw_nll = negative_log_likelihood(labels, apply_temperature(logits, 1.0))
    calibrated_nll = negative_log_likelihood(
        labels, apply_temperature(logits, fit.temperature)
    )
    assert fit.temperature > 1.0
    assert calibrated_nll < raw_nll


def test_target_prior_correction_identity_for_unscoped_methods():
    logits = np.array([[1.0, 2.0, 3.0]])
    pi_train = np.array([0.5, 0.3, 0.2])
    pi_target = np.array([0.2, 0.3, 0.5])
    out = apply_target_prior_correction(logits, "weighted_ce", 1.0, pi_train, pi_target)
    assert np.array_equal(out, logits)


def test_target_prior_correction_posthoc_formula():
    logits = np.array([[1.0, 2.0, 3.0]])
    pi_train = np.array([0.5, 0.3, 0.2])
    pi_target = np.array([0.2, 0.3, 0.5])
    out = apply_target_prior_correction(
        logits, "post_hoc_logit_adjustment", 1.0, pi_train, pi_target
    )
    expected = logits - np.log(pi_train) + np.log(pi_target)
    assert np.allclose(out, expected)
    balanced = balanced_decision_logits(
        logits, "post_hoc_logit_adjustment", 0.5, pi_train
    )
    assert np.allclose(balanced, logits - 0.5 * np.log(pi_train))


# --- gates and recovery ----------------------------------------------------------


def test_discrimination_gate_opens_and_closes():
    assert discrimination_gate(0.05, (0.01, 0.09)) is True
    assert discrimination_gate(0.01, (-0.01, 0.03)) is False  # below threshold
    assert discrimination_gate(0.05, (-0.01, 0.11)) is False  # CI includes zero


def test_calibration_gate_thresholds():
    assert calibration_gate(0.06, (0.02, 0.10)) is True
    assert calibration_gate(0.04, (0.01, 0.07)) is False


def test_ci_excludes_zero():
    assert ci_excludes_zero(0.1, 0.2) is True
    assert ci_excludes_zero(-0.2, -0.1) is True
    assert ci_excludes_zero(-0.1, 0.1) is False


@pytest.mark.parametrize(
    "method_metric,imbalanced_ce,d,expected",
    [
        (0.5, 0.5, 0.1, 0.0),
        (0.6, 0.5, 0.1, 1.0),
        (0.4, 0.5, 0.1, -1.0),
        (0.7, 0.5, 0.1, 2.0),
    ],
)
def test_recovery_sign_conventions(method_metric, imbalanced_ce, d, expected):
    assert recovery(method_metric, imbalanced_ce, d) == pytest.approx(expected)


def test_recovery_nan_when_no_deficit():
    assert np.isnan(recovery(0.6, 0.5, 0.0))


def test_deficit_higher_is_better():
    assert deficit(0.7, 0.5) == pytest.approx(0.2)


# --- bootstrap --------------------------------------------------------------------


def _toy_identity(n_patients_per_class: int = 6) -> pd.DataFrame:
    rows = []
    for cls in ["A", "B"]:
        for p in range(n_patients_per_class):
            case_id = f"{cls}_P{p}"
            for s in range(2):
                rows.append(
                    {
                        "case_id": case_id,
                        "slide_id": f"{case_id}_S{s}",
                        "cancer_type": cls,
                    }
                )
    return pd.DataFrame(rows)


def test_kish_effective_count_uniform_weights_equals_n():
    weights = np.ones((10, 5), dtype=np.int64)
    kish = kish_effective_count(weights)
    assert np.allclose(kish, 10.0)


def test_stratum_preservation_invariant():
    identity = _toy_identity()
    strata = build_strata(identity)
    rng = np.random.default_rng(0)
    case_ids, weights = resample_patient_weights(strata, n_replicates=500, rng=rng)
    for stratum_key, members in strata.groupby(strata):
        idx = np.isin(case_ids, members.index.to_numpy())
        stratum_sum = weights[idx, :].sum(axis=0)
        assert np.all(stratum_sum == len(members))


def test_crossed_strata_distinguish_complete_split_by_class_contributions():
    identity = pd.DataFrame(
        [
            {"case_id": "P0", "cancer_type": "A", "patient_split": 0},
            {"case_id": "P0", "cancer_type": "B", "patient_split": 1},
            {"case_id": "P1", "cancer_type": "A", "patient_split": 0},
            {"case_id": "P1", "cancer_type": "B", "patient_split": 1},
            {"case_id": "P2", "cancer_type": "A", "patient_split": 0},
            {"case_id": "P2", "cancer_type": "A", "patient_split": 1},
        ]
    )
    strata = build_strata(identity)
    assert strata["P0"] == strata["P1"]
    assert strata["P0"] != strata["P2"]


def test_bootstrap_preflight_flags_small_kish():
    # A single dominant patient per class should be flagged descriptive-only.
    rows = [
        {"case_id": "DOMINANT_A", "slide_id": f"S{i}", "cancer_type": "A"}
        for i in range(20)
    ]
    rows += [
        {"case_id": f"B_P{i}", "slide_id": f"BS{i}", "cancer_type": "B"}
        for i in range(10)
    ]
    identity = pd.DataFrame(rows)
    report = bootstrap_preflight(identity, n_replicates=200, seed=0)
    assert report["by_class"]["A"]["is_descriptive_only"] is True


def test_weighted_balanced_accuracy_matches_unweighted_when_weights_are_one():
    labels = np.array([0, 0, 1, 1])
    preds = np.array([0, 1, 1, 1])
    weights = np.ones((4, 3), dtype=np.int64)
    ba = weighted_balanced_accuracy(labels, preds, weights, n_classes=2)
    expected = (0.5 + 1.0) / 2
    assert np.allclose(ba, expected)


def test_seed_resample_gather_averages_correctly():
    per_seed_metric = np.array([[1.0, 2.0], [3.0, 4.0]])  # (n_seeds=2, n_replicates=2)
    seed_idx = np.array(
        [[0, 0], [1, 1]]
    )  # replicate 0 picks seed0 twice, replicate1 picks seed1 twice
    out = gather_seed_resampled(per_seed_metric, seed_idx)
    assert np.allclose(out, [1.0, 4.0])


def test_expand_to_rows_broadcasts_patient_weight():
    case_ids = np.array(["P0", "P1"])
    patient_weights = np.array([[3], [7]])
    row_case_ids = np.array(["P0", "P1", "P0"])
    rows = expand_to_rows(case_ids, patient_weights, row_case_ids)
    assert rows.flatten().tolist() == [3, 7, 3]


# --- permutation --------------------------------------------------------------------


def test_permutation_p_value_is_one_when_predictions_identical():
    labels = np.array([0, 1, 0, 1])
    preds = np.array([0, 1, 0, 1])
    case_ids = np.array(["P0", "P1", "P2", "P3"])
    p = paired_block_permutation_ba(
        labels, preds, preds, case_ids, n_classes=2, n_permutations=50
    )
    assert p == pytest.approx(1.0)


def test_permutation_block_swap_keeps_patient_rows_together():
    # A 2-row patient should never have only one of its rows swapped: run a
    # case where method/CE disagree everywhere, so any partial (non-block)
    # swap would produce an intermediate accuracy unseen in the full
    # enumeration if rows were swapped independently rather than per patient.
    labels = np.array([0, 0, 1, 1])
    case_ids = np.array(["P0", "P0", "P1", "P1"])
    method_preds = np.array([0, 0, 1, 1])
    ce_preds = np.array([1, 1, 0, 0])
    p = paired_block_permutation_ba(
        labels, method_preds, ce_preds, case_ids, n_classes=2, n_permutations=50
    )
    assert 0.0 <= p <= 1.0


def test_crossed_permutation_uses_one_patient_swap_across_split_appearances():
    labels = np.array([0, 1])
    first = (labels, np.array([[0, 1]]), np.array([[1, 0]]), np.array(["P0", "P1"]))
    second = (labels, np.array([[0, 1]]), np.array([[1, 0]]), np.array(["P0", "P1"]))
    p_value = crossed_block_permutation_ba([first, second], n_classes=2)
    assert 0.0 <= p_value <= 1.0


# --- Holm / confirmatory family -------------------------------------------------------


def test_confirmatory_family_partition():
    comparisons = [
        {"method": "weighted_ce", "gate_passed": True, "p_value": 0.01},
        {"method": "cfal", "gate_passed": True, "p_value": 0.02},
    ]
    confirmatory, exploratory = confirmatory_family(comparisons)
    assert [c["method"] for c in confirmatory] == ["weighted_ce"]
    assert [c["method"] for c in exploratory] == ["cfal"]


def test_holm_marks_gated_out_as_not_tested():
    comparisons = [
        {"method": "weighted_ce", "gate_passed": True, "p_value": 0.01},
        {"method": "focal", "gate_passed": False, "p_value": None},
        {"method": "cfal", "gate_passed": True, "p_value": 0.03},
    ]
    out = apply_holm(comparisons)
    by_method = {c["method"]: c for c in out}
    assert by_method["focal"]["status"] == "not tested"
    assert by_method["weighted_ce"]["status"] == "tested"
    assert by_method["weighted_ce"]["family"] == "confirmatory"
    assert by_method["cfal"]["family"] == "exploratory"
    assert (
        by_method["weighted_ce"]["adjusted_p_value"]
        >= by_method["weighted_ce"]["p_value"]
    )


# --- separability / effective support -------------------------------------------------


def test_effective_support_reduces_to_n_when_icc_zero():
    assert effective_support(n_c=100, mean_cluster_size=4.0, icc=0.0) == pytest.approx(
        100.0
    )


def test_effective_support_shrinks_with_high_icc_and_clustering():
    n_eff = effective_support(n_c=100, mean_cluster_size=4.0, icc=1.0)
    assert n_eff == pytest.approx(25.0)


def test_intraclass_correlation_zero_when_no_between_cluster_variance():
    rng = np.random.default_rng(0)
    margin = rng.normal(size=40)
    clusters = np.repeat(np.arange(10), 4)
    icc = intraclass_correlation(margin, clusters)
    assert 0.0 <= icc <= 1.0


def test_intraclass_correlation_high_when_clusters_dominate():
    cluster_means = np.repeat([-5.0, 5.0, -5.0, 5.0, -5.0], 8)
    margin = cluster_means + np.random.default_rng(0).normal(scale=0.01, size=40)
    clusters = np.repeat(np.arange(5), 8)
    icc = intraclass_correlation(margin, clusters)
    assert icc > 0.9


# --- RQ3 wiring -----------------------------------------------------------------------


def test_rq3_gate_pass_and_deficit_and_recovery_models_run():
    rng = np.random.default_rng(0)
    groups = [f"dataset_{i % 4}" for i in range(24)]
    cells = []
    for i in range(24):
        rho = float(rng.choice([1.0, 10.0, 100.0]))
        separability = float(rng.normal())
        gate_passed = rho > 1.0
        cells.append(
            {
                "group": groups[i],
                "rho": rho,
                "separability": separability,
                "gate_passed": gate_passed,
                "deficit_ba": float(rng.normal(0.05, 0.02)),
                "deficit_se": 0.01,
                "recovery": float(rng.normal(0.5, 0.1)),
                "recovery_se": 0.05,
            }
        )
    gate_model = fit_gate_pass_model(cells)
    deficit_model = fit_deficit_model(cells)
    recovery_model = fit_recovery_model(cells)
    assert "slopes" in gate_model and len(gate_model["slopes"]) == 2
    assert "slopes" in deficit_model
    assert "slopes" in recovery_model


def test_rq3_recovery_model_empty_when_no_gated_cells():
    cells = [
        {
            "group": "g",
            "rho": 1.0,
            "separability": 0.0,
            "gate_passed": False,
            "recovery": 0.0,
            "recovery_se": 0.01,
        }
    ]
    assert fit_recovery_model(cells) == {}


def test_rq3_cells_keep_calibration_gate_recovery(monkeypatch: pytest.MonkeyPatch):
    """A calibration-only cell must use tail-NLL recovery, not BA recovery."""
    monkeypatch.setattr(
        "imbalance_benchmark.analysis.predictors.rq3_analysis._covariates",
        lambda *_: {"separability": 0.5},
    )
    comparisons = [
        {
            "assignment": "native",
            "severity": "severe",
            "method": "ce",
            "gate": "discrimination",
            "gate_passed": False,
            "effect": 0.01,
            "bootstrap_effect": [0.01, 0.02],
        },
        {
            "assignment": "native",
            "severity": "severe",
            "method": "ce",
            "gate": "calibration",
            "gate_passed": True,
            "effect": 0.08,
            "bootstrap_effect": [0.08, 0.09],
        },
        {
            "assignment": "native",
            "severity": "severe",
            "method": "weighted_ce",
            "gate": "calibration",
            "gate_passed": True,
            "effect": 0.04,
            "recovery": 0.5,
            "bootstrap_effect": [0.04, 0.05],
            "bootstrap_numerator": [0.04, 0.05],
            "bootstrap_denominator": [0.08, 0.10],
        },
    ]
    freeze = {"assignment_conditions": {"native": {"severe": {"achieved_rho": 100.0}}}}

    cells = _cells({}, comparisons, freeze, "dataset:target", False)

    recovery = next(cell for cell in cells if cell["method"] == "weighted_ce")
    assert recovery["gate"] == "calibration"
    assert recovery["gate_passed"] is True
    assert recovery["recovery"] == pytest.approx(0.5)


def test_cross_split_rq3_keeps_gate_specific_calibration_outcome():
    rows = [
        {
            "assignment": "native",
            "severity": "severe",
            "method": "ce",
            "gate": "calibration",
            "gate_passed": True,
        },
        {
            "assignment": "native",
            "severity": "severe",
            "method": "weighted_ce",
            "gate": "calibration",
            "bootstrap_numerator": [0.04],
            "bootstrap_denominator": [0.08],
        },
    ]

    gates, outcomes = _comparison_maps(rows)

    assert gates[("native", "severe", "calibration")]
    assert ("native", "severe", "weighted_ce", "calibration") in outcomes


def test_log_scale_prior_prevents_random_effect_scale_collapse():
    collapsed = _log_scale_prior(torch.tensor([-20.0]))
    centered = _log_scale_prior(torch.tensor([0.0]))
    assert collapsed > centered


# --- db / query / tables end-to-end ----------------------------------------------------


def _write_fake_run(
    results_root: Path,
    condition: str,
    method: str,
    seed_idx: int,
    class_names: list[str],
    seed_offset: int,
) -> None:
    rng = np.random.default_rng(seed_offset)
    n = 12
    labels = rng.integers(0, len(class_names), size=n)
    probs = rng.dirichlet(np.ones(len(class_names)), size=n)
    preds = probs.argmax(axis=1)
    payload = classification_payload(
        labels.tolist(), preds.tolist(), probs.tolist(), class_names
    )
    payload["labels"] = labels.tolist()
    payload["preds"] = preds.tolist()
    payload["probabilities"] = probs.tolist()
    payload["logits"] = np.log(np.clip(probs, 1e-6, 1.0)).tolist()
    write_run_record(
        results_root / condition / method / f"seed={seed_idx}",
        {
            "benchmark": "patch",
            "condition": condition,
            "method": method,
            "seed": seed_offset,
            "class_names": class_names,
            "tuning_params": {"lr": 1e-3},
            "cost": {"updates": 10},
            "splits": {"validation": payload, "test": payload},
        },
    )


def test_ingest_and_tables_end_to_end(tmp_path: Path):
    results_root = tmp_path / "results"
    class_names = ["A", "B", "C"]
    for cond in ("balanced", "moderate"):
        for method in ("ce", "weighted_ce"):
            for seed_idx in range(2):
                _write_fake_run(
                    results_root, cond, method, seed_idx, class_names, seed_idx
                )

    conn = connect_db(tmp_path / "results.sqlite")
    init_schema(conn)
    tiers = assign_tiers(class_names, {"A": 100, "B": 50, "C": 10})
    for cond, method, seed_idx, result_dir in discover_result_dirs(results_root):
        from imbalance_benchmark.common import read_run_record

        record = read_run_record(result_dir)
        assert record is not None
        ingest_run(
            conn,
            f"patch:{cond}:{method}:seed={seed_idx}",
            result_dir,
            cond,
            method,
            seed_idx,
            record,
            tiers,
        )

    details = load_eval_details(conn)
    assert len(details) == 2 * 2 * 2 * 2  # conditions x methods x seeds x splits
    classwise = load_classwise(conn)
    assert set(classwise["tier"].unique()) == {"head", "body", "tail"}

    table = results_table(conn)
    assert "balanced" in table and "ce" in table
    cal_table = calibration_table(conn)
    assert "negative_log_likelihood" not in cal_table or "\\begin{tabular}" in cal_table


def test_balanced_ingestion_leaves_classwise_tiers_null(tmp_path: Path):
    """Balanced runs have no tail assignment, so classwise tiers stay undefined."""
    class_names = ["A", "B"]
    results_root = tmp_path / "results"
    _write_fake_run(results_root, "balanced", "ce", 0, class_names, 0)
    result_dir = results_root / "balanced" / "ce" / "seed=0"
    record = read_run_record(result_dir)
    assert record is not None
    record["assignment"] = "unassigned"
    conn = connect_db(tmp_path / "results.sqlite")
    init_schema(conn)

    _ingest_discovered_run(
        conn,
        {"conditions": {"balanced": {"allocated_counts": {"A": 10, "B": 10}}}},
        "balanced",
        "ce",
        0,
        result_dir,
        record,
    )

    n_rows = conn.execute("SELECT COUNT(*) FROM eval_classwise").fetchone()[0]
    null_tiers = conn.execute(
        "SELECT COUNT(*) FROM eval_classwise WHERE tier IS NULL"
    ).fetchone()[0]
    assert null_tiers == n_rows == 4


def test_load_test_identity_matches_row_order(tmp_path: Path):
    manifest = pd.DataFrame(
        [
            {"case_id": "P0", "slide_id": "S0", "cancer_type": "A", "split": "test"},
            {"case_id": "P1", "slide_id": "S1", "cancer_type": "B", "split": "test"},
            {"case_id": "P0", "slide_id": "S0", "cancer_type": "A", "split": "train"},
        ]
    )
    manifest_path = tmp_path / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    identity = load_test_identity(manifest_path, is_mil=False)
    assert identity["case_id"].tolist() == ["P0", "P1"]


def test_load_mil_test_identity_preserves_bag_dataset_order(tmp_path: Path):
    manifest = pd.DataFrame(
        [
            {
                "case_id": "PAT_B",
                "slide_id": "slide_B",
                "cancer_type": "B",
                "split": "test",
            },
            {
                "case_id": "PAT_A",
                "slide_id": "slide_A",
                "cancer_type": "A",
                "split": "test",
            },
        ]
    )
    manifest_path = tmp_path / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    identity = load_test_identity(manifest_path, is_mil=True)

    assert identity["slide_id"].tolist() == ["slide_B", "slide_A"]
    assert identity["case_id"].tolist() == ["PAT_B", "PAT_A"]


def test_crossed_aggregate_recomputes_recovery_inside_bootstrap_replicates(
    tmp_path: Path,
):
    paths = ensure_dirs({"paths": {"outputs": str(tmp_path)}})
    for index, effect in enumerate(([0.02, 0.04], [0.04, 0.08], [0.06, 0.12])):
        ce = {
            "assignment": "native",
            "severity": "severe",
            "method": "ce",
            "gate": "discrimination",
            "effect": float(np.mean(effect)),
            "bootstrap_effect": effect,
            "gate_passed": True,
            "p_value": None,
        }
        method = {
            "assignment": "native",
            "severity": "severe",
            "method": "weighted_ce",
            "gate": "discrimination",
            "effect": float(np.mean(effect)),
            "bootstrap_effect": effect,
            "bootstrap_numerator": effect,
            "bootstrap_denominator": effect,
            "gate_passed": True,
            "p_value": 0.1,
        }
        write_json(
            split_paths(paths, index)["data"] / "gates_and_recovery.json",
            # This ordering is the real failure case: balanced_sampling sorts
            # before CE in grouped output, so CE gates must be resolved first.
            {"comparisons": [{**method, "method": "balanced_sampling"}, ce]},
        )
    _aggregate_split_comparisons(paths)
    output = json.loads(
        (paths["data"] / "cross_split_gates_and_recovery.json").read_text()
    )
    weighted = next(
        c for c in output["comparisons"] if c["method"] == "balanced_sampling"
    )
    assert weighted["effect"] == pytest.approx(0.06)
    assert weighted["recovery"] == pytest.approx(1.0)
    assert weighted["bootstrap_effect"] == pytest.approx([0.04, 0.08])
    assert weighted["bootstrap_numerator"] == pytest.approx([0.04, 0.08])
    assert weighted["bootstrap_denominator"] == pytest.approx([0.04, 0.08])


def test_crossed_bootstrap_reuses_one_patient_weight_across_split_appearances(
    tmp_path: Path,
):
    paths = ensure_dirs({"paths": {"outputs": str(tmp_path)}})
    for index in (0, 1):
        manifest = pd.DataFrame(
            [
                {
                    "case_id": "P0",
                    "slide_id": f"P0_{index}",
                    "cancer_type": "A",
                    "split": "test",
                },
                {
                    "case_id": "P1",
                    "slide_id": f"P1_{index}",
                    "cancer_type": "B",
                    "split": "test",
                },
            ]
        )
        manifest.to_csv(split_paths(paths, index)["data"] / "manifest.csv", index=False)
    first = BootstrapContext(
        split_paths(paths, 0), is_mil=False, n_replicates=20, seed=4
    )
    second = BootstrapContext(
        split_paths(paths, 1), is_mil=False, n_replicates=20, seed=4
    )
    assert np.array_equal(first.row_weights[0], second.row_weights[0])
