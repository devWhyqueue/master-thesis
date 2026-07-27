from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from imbalance_benchmark.analysis.aggregation.aggregate import (
    require_complete_split_comparisons,
    require_consistent_achieved_severity,
)
from imbalance_benchmark.analysis.db import (
    connect_db,
    discover_result_dirs,
    ingest_run,
    init_schema,
)
from imbalance_benchmark.analysis.inference.bootstrap import (
    PatientWeights,
    build_strata,
    expand_to_rows,
    gather_seed_resampled,
    kish_effective_count,
    resample_patient_weights,
    weighted_balanced_accuracy,
    weighted_ece,
    weighted_macro_nll,
)
from imbalance_benchmark.analysis.inference.context import BootstrapContext
from imbalance_benchmark.analysis.inference.gates import (
    ci_excludes_zero,
)
from imbalance_benchmark.analysis.inference.preflight import bootstrap_preflight
from imbalance_benchmark.analysis.metrics import (
    assign_tiers,
    classification_payload,
    negative_log_likelihood,
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

def test_ci_excludes_zero():
    assert ci_excludes_zero(0.1, 0.2) is True
    assert ci_excludes_zero(-0.2, -0.1) is True
    assert ci_excludes_zero(-0.1, 0.1) is False

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
    weights = PatientWeights(np.arange(4), np.ones((4, 3), dtype=np.float64))
    ba = weighted_balanced_accuracy(labels, preds, weights, n_classes=2)
    expected = (0.5 + 1.0) / 2
    assert np.allclose(ba, expected)

def test_patient_weights_kernels_match_naive_row_expansion():
    """The patient-collapsed kernels must equal the same statistic computed by
    naively expanding each patient's weight to every one of its rows first --
    this is the check that fails if the algebraic collapse is ever wrong.
    """
    rng = np.random.default_rng(0)
    n_patients, n_replicates, n_classes = 6, 5, 3
    rows_per_patient = rng.integers(1, 4, size=n_patients)
    row_patient = np.repeat(np.arange(n_patients), rows_per_patient)
    n_rows = len(row_patient)
    patient_weights = rng.integers(0, 5, size=(n_patients, n_replicates)).astype(
        np.float64
    )
    weights = PatientWeights(row_patient, patient_weights)
    row_weights = patient_weights[row_patient]  # naive (n_rows, n_replicates) expansion

    labels = rng.integers(0, n_classes, size=n_rows)
    preds = rng.integers(0, n_classes, size=n_rows)
    probs = rng.dirichlet(np.ones(n_classes), size=n_rows)

    def naive_balanced_accuracy() -> np.ndarray:
        out = np.zeros(n_replicates)
        for c in range(n_classes):
            mask = labels == c
            if not mask.any():
                continue
            correct = mask & (preds == c)
            class_weight = row_weights[mask].sum(axis=0)
            correct_weight = row_weights[correct].sum(axis=0)
            out += np.where(
                class_weight > 0, correct_weight / np.maximum(class_weight, 1e-12), 0.0
            )
        return out / n_classes

    def naive_macro_nll(class_subset: list[int]) -> np.ndarray:
        out = np.zeros(n_replicates)
        per_sample_nll = -np.log(np.clip(probs[np.arange(n_rows), labels], 1e-12, 1.0))
        counted = 0
        for c in class_subset:
            mask = labels == c
            if not mask.any():
                continue
            w = row_weights[mask]
            class_weight = w.sum(axis=0)
            weighted_nll = (w * per_sample_nll[mask, None]).sum(axis=0)
            out += np.where(
                class_weight > 0, weighted_nll / np.maximum(class_weight, 1e-12), 0.0
            )
            counted += 1
        return out / max(counted, 1)

    def naive_ece(n_bins: int = 10) -> np.ndarray:
        confidence = probs.max(axis=1)
        correct = (probs.argmax(axis=1) == labels).astype(np.float64)
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        bin_of_row = np.clip(np.digitize(confidence, edges[1:-1]), 0, n_bins - 1)
        total = row_weights.sum(axis=0)
        out = np.zeros(n_replicates)
        for b in range(n_bins):
            mask = bin_of_row == b
            if not mask.any():
                continue
            w = row_weights[mask]
            bin_weight = w.sum(axis=0)
            acc = (w * correct[mask, None]).sum(axis=0)
            conf = (w * confidence[mask, None]).sum(axis=0)
            gap = np.where(
                bin_weight > 0, np.abs(acc - conf) / np.maximum(bin_weight, 1e-12), 0.0
            )
            out += np.where(total > 0, gap * bin_weight / np.maximum(total, 1e-12), 0.0)
        return out

    assert np.allclose(
        weighted_balanced_accuracy(labels, preds, weights, n_classes),
        naive_balanced_accuracy(),
    )
    assert np.allclose(
        weighted_macro_nll(labels, probs, weights, list(range(n_classes))),
        naive_macro_nll(list(range(n_classes))),
    )
    assert np.allclose(weighted_ece(labels, probs, weights), naive_ece())

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
            (cond, method, seed_idx),
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
            "effect": float(effect[0]),
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
    # Replicate 0 is the observed cross-split effect: mean(0.02, 0.04, 0.06).
    assert weighted["effect"] == pytest.approx(0.04)
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
    first_patient_of_row0 = first.weights.row_patient[0]
    second_patient_of_row0 = second.weights.row_patient[0]
    assert np.array_equal(
        first.weights.patient[first_patient_of_row0],
        second.weights.patient[second_patient_of_row0],
    )

def test_cross_split_completeness_rejects_a_roster_method_missing_everywhere() -> None:
    rows = [
        {
            "patient_split": split,
            "assignment": "native",
            "severity": "severe",
            "method": "ce",
            "gate": "discrimination",
        }
        for split in range(3)
    ]
    expected = {
        ("native", "severe", "ce", "discrimination"),
        ("native", "severe", "focal", "discrimination"),
    }

    with pytest.raises(RuntimeError, match="missing"):
        require_complete_split_comparisons(rows, expected)

def test_recovery_standard_error_uses_the_recovery_distribution() -> None:
    """recovery_se must come from numerator/denominator, not the raw effect spread."""
    from imbalance_benchmark.analysis.predictors.rq3_analysis import (
        _recovery_standard_error,
        _standard_error,
    )

    rng = np.random.default_rng(1)
    numerator = rng.normal(0.3, 0.05, size=500)
    denominator = rng.normal(0.5, 0.05, size=500)
    comparison = {
        "bootstrap_effect": numerator.tolist(),
        "bootstrap_numerator": numerator.tolist(),
        "bootstrap_denominator": denominator.tolist(),
    }

    recovery_se = _recovery_standard_error(comparison)
    effect_se = _standard_error(comparison)

    expected = float(np.nanstd(numerator / denominator, ddof=1))
    assert recovery_se == pytest.approx(expected)
    assert recovery_se != pytest.approx(effect_se)

def test_require_consistent_achieved_severity_rejects_one_null_split_among_two_real(
    tmp_path: Path,
) -> None:
    """Two real splits and one collapsed-to-balanced split must not be averaged."""
    paths = ensure_dirs({"paths": {"outputs": str(tmp_path)}})
    for index, achieved in enumerate([1.0, 10.0, 10.0]):
        write_json(
            split_paths(paths, index)["data"] / "manifest_freeze.json",
            {
                "assignment_conditions": {
                    "native": {
                        "severe": {"achieved_rho": achieved, "allocated_counts": {}}
                    }
                }
            },
        )

    with pytest.raises(RuntimeError, match="differs materially"):
        require_consistent_achieved_severity(paths)

def test_require_consistent_achieved_severity_allows_modest_natural_variation(
    tmp_path: Path,
) -> None:
    paths = ensure_dirs({"paths": {"outputs": str(tmp_path)}})
    for index, achieved in enumerate([9.2, 10.0, 9.8]):
        write_json(
            split_paths(paths, index)["data"] / "manifest_freeze.json",
            {
                "assignment_conditions": {
                    "native": {
                        "severe": {"achieved_rho": achieved, "allocated_counts": {}}
                    }
                }
            },
        )

    require_consistent_achieved_severity(paths)

def test_cross_split_aggregation_requires_every_comparison_in_all_three_splits() -> (
    None
):
    rows = [
        {
            "patient_split": 0,
            "assignment": "native",
            "severity": "severe",
            "method": "ce",
            "gate": "discrimination",
        },
        {
            "patient_split": 1,
            "assignment": "native",
            "severity": "severe",
            "method": "ce",
            "gate": "discrimination",
        },
        {
            "patient_split": 2,
            "assignment": "native",
            "severity": "severe",
            "method": "ce",
            "gate": "discrimination",
        },
        {
            "patient_split": 0,
            "assignment": "native",
            "severity": "severe",
            "method": "focal",
            "gate": "discrimination",
        },
    ]

    with pytest.raises(RuntimeError, match="incomplete"):
        require_complete_split_comparisons(rows)
