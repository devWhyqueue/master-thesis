from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from imbalance_benchmark.analysis.aggregate import require_complete_split_comparisons
from imbalance_benchmark.analysis.reporting.ingestion import _run_calibration
from imbalance_benchmark.commands.confirm import require_tuning_configs
from imbalance_benchmark.commands.freeze_execution import wsi_bootstrap_identity
from imbalance_benchmark.datasets.data import BagFeatureDataset, ImbalanceDataset
from imbalance_benchmark.modeling.context import cost_payload
from imbalance_benchmark.modeling.training import _fit_step


def test_cfal_tuned_sigma_is_not_reused_as_a_loss_margin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CFAL's only method-specific grid control is its affinity bandwidth."""
    seen: dict[str, object] = {}

    def fixed_loss(
        model: torch.nn.Module,
        features: torch.Tensor,
        targets: torch.Tensor,
        class_counts: np.ndarray,
    ) -> torch.Tensor:
        seen["called"] = True
        return model(features).sum()

    monkeypatch.setattr("imbalance_benchmark.modeling.training.cfal_loss", fixed_loss)
    model = torch.nn.Linear(2, 2)
    loss = _fit_step(
        {"features": torch.ones(2, 2), "target": torch.tensor([0, 1])},
        {
            "is_mil": False,
            "method": "cfal",
            "device": torch.device("cpu"),
            "model": model,
            "criterion": torch.nn.CrossEntropyLoss(),
            "param": 4.0,
            "class_counts": np.array([1, 1]),
        },
        0,
        1,
    )

    assert seen["called"] is True
    assert loss.requires_grad


def test_wsi_preflight_uses_one_identity_row_per_slide() -> None:
    raw = pd.DataFrame(
        [
            {"case_id": "p1", "slide_id": "s1", "cancer_type": "A"},
            {"case_id": "p1", "slide_id": "s1", "cancer_type": "A"},
            {"case_id": "p2", "slide_id": "s2", "cancer_type": "B"},
        ]
    )

    identity = wsi_bootstrap_identity(raw)

    assert identity[["case_id", "slide_id", "cancer_type"]].to_dict("records") == [
        {"case_id": "p1", "slide_id": "s1", "cancer_type": "A"},
        {"case_id": "p2", "slide_id": "s2", "cancer_type": "B"},
    ]


def test_dataset_uses_the_locked_global_class_index_even_when_a_split_is_sparse(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.csv"
    feature = tmp_path / "feature.pt"
    torch.save(torch.ones(1, 2), feature)
    pd.DataFrame(
        [
            {
                "case_id": "p1",
                "slide_id": "s1",
                "cancer_type": "A",
                "feature_path": feature,
                "split": "validation",
            }
        ]
    ).to_csv(manifest, index=False)

    dataset = ImbalanceDataset(manifest, "validation", class_names=["A", "B"])

    assert dataset.classes == ["A", "B"]
    assert dataset.get_n_classes() == 2
    assert dataset[0]["target"] == 0


def test_missing_tuning_selection_stops_confirmation() -> None:
    with pytest.raises(RuntimeError, match="missing tuning selection"):
        require_tuning_configs({"ce": {"lr": 1e-3}}, ("ce", "weighted_ce"))


def test_bag_dataset_rejects_multiple_labels_for_one_slide(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {
                "case_id": "p1",
                "slide_id": "s1",
                "cancer_type": "N",
                "feature_path": "unused.pt",
            },
            {
                "case_id": "p1",
                "slide_id": "s1",
                "cancer_type": "IC",
                "feature_path": "unused.pt",
            },
        ]
    ).to_csv(manifest, index=False)

    with pytest.raises(ValueError, match="exactly one class"):
        BagFeatureDataset(manifest)


def test_calibration_summary_retains_scaled_outputs_and_all_claimed_metrics() -> None:
    record = {
        "splits": {
            "validation": {"labels": [0, 1], "logits": [[4.0, 0.0], [0.0, 4.0]]},
            "test": {
                "labels": [0, 1],
                "logits": [[2.0, 0.0], [0.0, 2.0]],
                "probabilities": [[0.88, 0.12], [0.12, 0.88]],
            },
        }
    }

    summary = _run_calibration(record)

    assert summary is not None
    assert {"temperature_scaled_logits", "temperature_scaled_probabilities"} <= set(summary)
    assert {"temperature_scaled_test_nll", "temperature_scaled_test_brier", "temperature_scaled_test_ece"} <= set(summary)
    assert "temperature_scaled_reliability" in summary


def test_cost_uses_actual_processed_examples_for_partial_batches() -> None:
    cost = cost_payload(
        "ce",
        budget=3,
        elapsed=1.0,
        model=torch.nn.Linear(2, 2),
        unique_examples=10,
        exposed_examples=10,
        processed_examples=10,
    )

    assert cost["processed_examples"] == 10
    assert cost["effective_passes_through_unique_examples"] == 1.0


def test_cross_split_aggregation_requires_every_comparison_in_all_three_splits() -> None:
    rows = [
        {"patient_split": 0, "assignment": "native", "severity": "severe", "method": "ce", "gate": "discrimination"},
        {"patient_split": 1, "assignment": "native", "severity": "severe", "method": "ce", "gate": "discrimination"},
        {"patient_split": 2, "assignment": "native", "severity": "severe", "method": "ce", "gate": "discrimination"},
        {"patient_split": 0, "assignment": "native", "severity": "severe", "method": "focal", "gate": "discrimination"},
    ]

    with pytest.raises(RuntimeError, match="incomplete"):
        require_complete_split_comparisons(rows)
