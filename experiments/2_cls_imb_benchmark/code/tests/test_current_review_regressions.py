from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from imbalance_benchmark.analysis.aggregate import require_complete_split_comparisons
from imbalance_benchmark.analysis.calibration import (
    seed_averaged_reliability_curve,
    temperature_scaled_payload,
)
from imbalance_benchmark.analysis.inference.bootstrap import _class_preflight
from imbalance_benchmark.analysis.reporting.clustered_endpoints import clustered_endpoints
from imbalance_benchmark.analysis.reporting.plots import (
    allocated_training_support,
    plot_tail_vs_support,
)
from imbalance_benchmark.construction import max_shared_total
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
        [{"assignment": "native", "condition": "severe", "class_name": "A", "support": 99}]
    )
    freeze = {
        "assignment_conditions": {
            "native": {"severe": {"allocated_counts": {"A": 7}}}
        }
    }

    assert allocated_training_support(classwise, freeze).tolist() == [7]


def test_tail_support_plot_excludes_conditions_without_a_frozen_tier(
    tmp_path: Path,
) -> None:
    classwise = pd.DataFrame(
        [
            {"assignment": "native", "condition": "natural", "class_name": "A", "tier": None, "support": 99, "recall": 0.5},
            {"assignment": "native", "condition": "severe", "class_name": "A", "tier": "tail", "support": 99, "recall": 0.5},
        ]
    )
    freeze = {"assignment_conditions": {"native": {"severe": {"allocated_counts": {"A": 7}}}}}

    plot_tail_vs_support(classwise, freeze, tmp_path / "tail.png")

    assert (tmp_path / "tail.png").exists()


def test_cluster_macro_balanced_accuracy_is_macro_recall_after_cluster_aggregation() -> None:
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
