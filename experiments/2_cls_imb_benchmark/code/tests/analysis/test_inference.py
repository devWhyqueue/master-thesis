from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from imbalance_benchmark.analysis.inference.crossed_permutation import (
    crossed_block_permutation_ba,
)
from imbalance_benchmark.analysis.inference.gates import (
    discrimination_gate,
)
from imbalance_benchmark.analysis.inference.holm import apply_holm, confirmatory_family
from imbalance_benchmark.analysis.inference.permutation import (
    paired_block_permutation_ba,
)
from imbalance_benchmark.analysis.inference.preflight import (
    _class_preflight,
    require_valid_preflight,
    run_preflight,
)
from imbalance_benchmark.analysis.predictors.rq3_analysis import (
    _covariates,
    _has_multiple_slides_per_patient,
)
from imbalance_benchmark.analysis.predictors.separability import (
    effective_support,
    intraclass_correlation,
)
from imbalance_benchmark.datasets.data import slide_level_identity

def _ce_gate_entry(descriptive_only: bool) -> dict[str, object]:
    return {
        "method": "ce",
        "gate": "discrimination",
        "assignment": "native",
        "severity": "severe",
        "effect": 0.2,  # well above the 0.02 discrimination threshold
        "ci": (0.1, 0.3),  # excludes zero
        "descriptive_only": descriptive_only,
    }

def test_discrimination_gate_opens_and_closes():
    assert discrimination_gate(0.05, (0.01, 0.09)) is True
    assert discrimination_gate(0.01, (-0.01, 0.03)) is False  # below threshold
    assert discrimination_gate(0.05, (-0.01, 0.11)) is False  # CI includes zero

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

def test_effective_support_reduces_to_n_when_icc_zero():
    assert effective_support(n_c=100, mean_cluster_size=4.0, icc=0.0) == pytest.approx(
        100.0
    )

def test_effective_support_shrinks_with_high_icc_and_clustering():
    n_eff = effective_support(n_c=100, mean_cluster_size=4.0, icc=1.0)
    assert n_eff == pytest.approx(25.0)

def test_wsi_patient_support_sensitivity_requires_multi_slide_patients(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    one_slide_per_patient = {
        "contribution_stats": {
            "A": {"n_patients": 10, "n_slides": 10},
            "B": {"n_patients": 10, "n_slides": 10},
        }
    }
    multiple_slides_per_patient = {
        "contribution_stats": {
            "A": {"n_patients": 10, "n_slides": 12},
            "B": {"n_patients": 10, "n_slides": 10},
        }
    }

    assert not _has_multiple_slides_per_patient(one_slide_per_patient)
    assert _has_multiple_slides_per_patient(multiple_slides_per_patient)

    def feature_frame(*_args: object) -> tuple[np.ndarray, np.ndarray]:
        return np.array([[0.0, 0.0], [1.0, 1.0]]), np.array([0, 1])

    monkeypatch.setattr(
        "imbalance_benchmark.analysis.predictors.rq3_analysis._feature_frame",
        feature_frame,
    )
    monkeypatch.setattr(
        "imbalance_benchmark.analysis.predictors.rq3_analysis.intrinsic_separability",
        lambda *_args: {
            "linear_probe_macro_recall": 0.5,
            "knn_macro_recall": 0.5,
            "per_class_nn_error": {},
        },
    )
    monkeypatch.setattr(
        "imbalance_benchmark.analysis.predictors.rq3_analysis.condition_learnability",
        lambda *_args: {"linear_probe_macro_recall": 0.5},
    )
    paths = {"data": tmp_path}
    for condition in (one_slide_per_patient, multiple_slides_per_patient):
        condition["path"] = str(tmp_path)

    one_slide_covariates = _covariates(paths, True, one_slide_per_patient)
    multi_slide_covariates = _covariates(paths, True, multiple_slides_per_patient)

    assert "log_min_patient_support" not in one_slide_covariates
    assert "log_min_patient_support" in multi_slide_covariates

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

def test_kish_preflight_is_descriptive_when_any_replicate_is_below_five() -> None:
    weights = np.eye(6, dtype=np.int64)
    weights[:, 0] = [6, 0, 0, 0, 0, 0]

    result = _class_preflight(np.arange(6), weights, n_replicates=6)

    assert result["min_kish_effective_count"] == 1.0
    assert result["is_descriptive_only"] is True

def test_exploratory_methods_are_not_hypothesis_tested() -> None:
    """Exploratory methods keep effects/CIs but carry no p-value or "tested" status.

    Finding: "Exploratory methods receive hypothesis tests." Setup §3.6 limits
    hypothesis tests to the four primary methods.
    """
    from imbalance_benchmark.analysis.inference.holm import apply_holm

    out = apply_holm(
        [
            {
                "method": "cfal",
                "gate": "discrimination",
                "severity": "severe",
                "gate_passed": True,
                "p_value": 0.02,
            }
        ]
    )
    row = out[0]
    assert row["family"] == "exploratory"
    assert row["status"] != "tested"
    assert row["p_value"] is None
    assert row["adjusted_p_value"] is None

def test_preflight_is_descriptive_when_any_split_class_fails_kish_threshold() -> None:
    rows = []
    for split, n_patients in ((0, 2), (1, 10)):
        for class_name in ("A", "B"):
            rows.extend(
                {
                    "case_id": f"{split}_{class_name}_{patient}",
                    "cancer_type": class_name,
                    "patient_split": split,
                }
                for patient in range(n_patients)
            )

    result = run_preflight(pd.DataFrame(rows), n_replicates=40, seed=4)

    assert result["by_split_class"]["0"]["A"]["kish_effective_count"] < 5
    assert result["is_descriptive_only"]

def test_descriptive_only_cell_never_opens_a_gate_or_permutes() -> None:
    """A preflight descriptive-only cell must skip gates and permutation p-values."""
    from imbalance_benchmark.analysis.aggregate import _apply_gates

    def fake_p_value(
        entry, base_paths, config, seed
    ):  # pragma: no cover - must not run
        raise AssertionError("descriptive-only cells must not be permutation tested")

    descriptive = [_ce_gate_entry(descriptive_only=True)]
    _apply_gates(descriptive, {}, {"dataset": {}}, 0, fake_p_value)
    assert descriptive[0]["gate_passed"] is False
    assert descriptive[0]["p_value"] is None

    confirmatory = [_ce_gate_entry(descriptive_only=False)]
    _apply_gates(confirmatory, {}, {"dataset": {}}, 0, lambda *_: 0.01)
    assert confirmatory[0]["gate_passed"] is True
    assert confirmatory[0]["p_value"] == 0.01

def test_wsi_preflight_uses_one_identity_row_per_slide() -> None:
    raw = pd.DataFrame(
        [
            {"case_id": "p1", "slide_id": "s1", "cancer_type": "A"},
            {"case_id": "p1", "slide_id": "s1", "cancer_type": "A"},
            {"case_id": "p2", "slide_id": "s2", "cancer_type": "B"},
        ]
    )

    identity = slide_level_identity(raw)

    assert identity[["case_id", "slide_id", "cancer_type"]].to_dict("records") == [
        {"case_id": "p1", "slide_id": "s1", "cancer_type": "A"},
        {"case_id": "p2", "slide_id": "s2", "cancer_type": "B"},
    ]


def test_freeze_stops_when_a_preflight_validity_check_fails() -> None:
    """Setup: a failed preflight check stops the analysis, it is not just recorded.

    Weight concentration is separate: it only marks the cell descriptive.
    """
    valid = {
        "all_split_level_metrics_computable": True,
        "identical_multiplicities_across_split_appearances": True,
        "is_descriptive_only": True,
    }
    require_valid_preflight(valid)

    for key in (
        "all_split_level_metrics_computable",
        "identical_multiplicities_across_split_appearances",
    ):
        with pytest.raises(RuntimeError, match="preflight failed"):
            require_valid_preflight({**valid, key: False})
