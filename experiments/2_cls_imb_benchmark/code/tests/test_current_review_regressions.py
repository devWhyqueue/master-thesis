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
from imbalance_benchmark.modeling.workflows import tuning_search


def test_single_split_post_hoc_uses_the_selected_ce_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The post-hoc checkpoint must belong to the winning CE configuration."""
    configs = [{"lr": 1.0}, {"lr": 2.0}]

    class Model:
        state: dict[str, float]

        def load_state_dict(self, state: dict[str, float]) -> None:
            self.state = state

    def context(*args: object) -> dict[str, object]:
        return {"model": Model(), "config": args[4], "seed": args[3]}

    def fit(ctx: dict[str, object]) -> tuple[dict[str, float], object]:
        return {"lr": float(ctx["config"]["lr"]), "seed": float(ctx["seed"])}, None

    def evaluate(model: Model, *args: object) -> dict[str, float]:
        return {
            "balanced_accuracy": 1.0 if model.state["lr"] == 1.0 else 0.0,
            "macro_f1": 0.0,
            "nll": 1.0,
        }

    monkeypatch.setattr(tuning_search, "get_grid_configs", lambda *_: configs)
    monkeypatch.setattr(tuning_search, "build_training_ctx", context)
    monkeypatch.setattr(tuning_search, "fit_method", fit)
    monkeypatch.setattr(tuning_search, "run_evaluation", evaluate)

    regime = type(
        "Regime", (), {"n_classes": 2, "device": None, "is_mil": False}
    )()
    selected, state = tuning_search._tune_grid("ce", None, None, regime, [0, 1])

    assert selected == {"lr": 1.0}
    assert state == {"lr": 1.0, "seed": 0.0}


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


def test_temperature_scaled_ece_has_a_patient_block_interval() -> None:
    payload = temperature_scaled_payload(
        np.array([[2.0, 0.0], [0.0, 2.0]]),
        np.array([0, 1]),
        np.array([[1.0, 0.0], [0.0, 1.0]]),
        np.array([0, 1]),
        np.array(["p1", "p2"]),
        seed=3,
    )

    assert len(payload["temperature_scaled_ece_ci"]) == 2


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

    endpoints = clustered_endpoints(labels, predictions, probabilities, identity, 0)

    assert endpoints["slide_macro_balanced_accuracy"] == pytest.approx(0.5)


def test_kish_preflight_is_descriptive_when_any_replicate_is_below_five() -> None:
    weights = np.eye(6, dtype=np.int64)
    weights[:, 0] = [6, 0, 0, 0, 0, 0]

    result = _class_preflight(np.arange(6), weights, n_replicates=6)

    assert result["min_kish_effective_count"] == 1.0
    assert result["is_descriptive_only"] is True


def test_balanced_reference_uses_the_largest_approximately_equal_total() -> None:
    assert max_shared_total([10, 10, 11], min_support=5) == 31
