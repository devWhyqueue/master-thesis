from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from imbalance_benchmark.analysis.aggregate import require_complete_split_comparisons
from imbalance_benchmark.analysis.calibration import (
    seed_averaged_reliability_curve,
    temperature_scaled_payload,
)
from imbalance_benchmark.analysis.inference.bootstrap import _class_preflight
from imbalance_benchmark.analysis.reporting.clustered_endpoints import (
    clustered_endpoints,
)
from imbalance_benchmark.analysis.reporting.plots import (
    allocated_training_support,
    plot_tail_vs_support,
)
from imbalance_benchmark.analysis.predictors.rq3_wiring import (
    fit_linked_sensitivity_models,
)
from imbalance_benchmark.construction import max_shared_total
from imbalance_benchmark.manifest.statistics import support_statistics


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


def test_temperature_scaled_ece_is_computed_without_a_per_run_bootstrap() -> None:
    payload = temperature_scaled_payload(
        np.array([[2.0, 0.0], [0.0, 2.0]]),
        np.array([0, 1]),
        np.array([[1.0, 0.0], [0.0, 1.0]]),
        np.array([0, 1]),
    )

    assert "temperature_scaled_ece_ci" not in payload


def test_reliability_bins_are_averaged_over_seeds_not_probabilities() -> None:
    probabilities = np.array(
        [
            [[0.9, 0.1], [0.1, 0.9]],
            [[0.6, 0.4], [0.6, 0.4]],
        ]
    )

    _, confidence, accuracy = seed_averaged_reliability_curve(
        probabilities, np.array([0, 1])
    )

    assert confidence.tolist() == pytest.approx([0.6, 0.9])
    assert accuracy.tolist() == pytest.approx([0.5, 1.0])


def test_tail_support_plot_uses_frozen_training_allocation() -> None:
    classwise = pd.DataFrame(
        [
            {
                "assignment": "native",
                "condition": "severe",
                "class_name": "A",
                "support": 99,
            }
        ]
    )
    freeze = {
        "assignment_conditions": {"native": {"severe": {"allocated_counts": {"A": 7}}}}
    }

    assert allocated_training_support(classwise, freeze).tolist() == [7]


def test_tail_support_plot_excludes_conditions_without_a_frozen_tier(
    tmp_path: Path,
) -> None:
    classwise = pd.DataFrame(
        [
            {
                "assignment": "native",
                "condition": "natural",
                "class_name": "A",
                "tier": None,
                "support": 99,
                "recall": 0.5,
            },
            {
                "assignment": "native",
                "condition": "severe",
                "class_name": "A",
                "tier": "tail",
                "support": 99,
                "recall": 0.5,
            },
        ]
    )
    freeze = {
        "assignment_conditions": {"native": {"severe": {"allocated_counts": {"A": 7}}}}
    }

    plot_tail_vs_support(classwise, freeze, tmp_path / "tail.png")

    assert (tmp_path / "tail.png").exists()


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


def test_kish_preflight_is_descriptive_when_any_replicate_is_below_five() -> None:
    weights = np.eye(6, dtype=np.int64)
    weights[:, 0] = [6, 0, 0, 0, 0, 0]

    result = _class_preflight(np.arange(6), weights, n_replicates=6)

    assert result["min_kish_effective_count"] == 1.0
    assert result["is_descriptive_only"] is True


def test_balanced_reference_uses_the_largest_approximately_equal_total() -> None:
    assert max_shared_total([10, 10, 11], min_support=5) == 31


def test_rq3_regime_specific_sensitivities_fit_their_eligible_cells(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Patch and WSI support sensitivities retain their respective regimes."""
    monkeypatch.setattr(
        "imbalance_benchmark.analysis.predictors.rq3_wiring.fit_rq3_model",
        lambda y, x, *_args, **_kwargs: {
            "n_observations": len(y),
            "values": x.ravel().tolist(),
        },
    )
    patch_cells = [
        {
            "group": "patch",
            "gate_passed": True,
            "deficit_ba": 0.1,
            "deficit_se": 0.01,
            "recovery": 0.5,
            "recovery_se": 0.02,
            "log_effective_support": value,
        }
        for value in (2.0, 3.0)
    ]
    wsi_cells = [
        {
            "group": "wsi",
            "gate_passed": True,
            "deficit_ba": 0.1,
            "deficit_se": 0.01,
            "recovery": 0.5,
            "recovery_se": 0.02,
            "log_min_patient_support": value,
        }
        for value in (4.0, 5.0)
    ]

    sensitivity = fit_linked_sensitivity_models(
        patch_cells + wsi_cells, patch_cells + wsi_cells
    )

    assert sensitivity["log_effective_support"]["deficit"]["values"] == [2.0, 3.0]
    assert sensitivity["log_min_patient_support"]["deficit"]["values"] == [4.0, 5.0]


def test_slide_statistics_count_mixed_label_slides_once_per_class() -> None:
    rows = pd.DataFrame(
        [
            {"slide_id": "s1", "cancer_type": "normal"},
            {"slide_id": "s1", "cancer_type": "tumor"},
            {"slide_id": "s2", "cancer_type": "normal"},
        ]
    )

    statistics = support_statistics(rows)

    assert statistics["patch"]["counts"] == {"normal": 2, "tumor": 1}
    assert statistics["slide"]["counts"] == {"normal": 2, "tumor": 1}
