from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import torch

from imbalance_benchmark.datasets import camelyon16
import imbalance_benchmark.datasets as dataset_adapters
from imbalance_benchmark.manifest.pilot_training import fit_pilot_model
from imbalance_benchmark.manifest.construction_helpers import write_natural_condition
from imbalance_benchmark.manifest.freeze import contribution_stats
from imbalance_benchmark.commands import tuning
from imbalance_benchmark.modeling.context import Regime
from imbalance_benchmark.modeling.context import cost_payload
from imbalance_benchmark.modeling.workflows.tuning_aggregate import (
    TuningScope,
    _select_trainable,
    summarize_tuning_cost,
)
from imbalance_benchmark.modeling.workflows.confirmation import (
    RunContext,
    confirm_method,
)


def test_train_time_logit_adjustment_confirmation_preserves_selected_tau(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Confirmation must evaluate train-time logit adjustment with its tuned tau."""
    observed: list[float] = []
    monkeypatch.setattr(
        "imbalance_benchmark.modeling.workflows.confirmation.build_training_ctx",
        lambda *_: {"seed": 7},
    )
    monkeypatch.setattr(
        "imbalance_benchmark.modeling.workflows.confirmation._timed_fit",
        lambda *_: ({}, 0.0),
    )
    monkeypatch.setattr(
        "imbalance_benchmark.modeling.workflows.confirmation._run_and_record",
        lambda *args: observed.append(float(args[7])),
    )
    run = RunContext(
        device=torch.device("cpu"),
        config={},
        n_classes=2,
        is_mil=False,
        class_names=["A", "B"],
        val_loader=object(),
        test_loader=object(),
        paths={"data": tmp_path, "results": tmp_path},
        seeds=[7],
        assignment="native",
    )

    confirm_method("severe", "logit_adjustment", {"parameter": 0.25}, object(), run)

    assert observed == [0.25]


def test_pilot_training_receives_the_configured_wsi_evidence_controls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The WSI pilot must use the same fixed instance cap and training config."""
    observed: dict[str, Any] = {}

    class Dataset:
        def __init__(self, *_: object, **kwargs: object) -> None:
            observed["bag_kwargs"] = kwargs

        def get_int_targets(self) -> torch.Tensor:
            return torch.tensor([0, 1])

        def __len__(self) -> int:
            return 2

    monkeypatch.setattr(
        "imbalance_benchmark.manifest.pilot_training.BagFeatureDataset", Dataset
    )

    def fit_model(ctx: dict[str, Any]) -> tuple[dict[str, Any], float]:
        observed["context"] = ctx
        return {}, 0.5

    monkeypatch.setattr(
        "imbalance_benchmark.manifest.pilot_training.fit_model", fit_model
    )

    fit_pilot_model(
        tmp_path / "pilot.csv",
        torch.device("cpu"),
        2,
        True,
        object(),
        initialization_seed=3,
        config={"wsi_training": {"max_instances": 17, "bag_batch_size": 2}},
        bag_kwargs={"max_instances": 17, "instance_selection_seed": 11},
    )

    assert observed["bag_kwargs"] == {
        "device": torch.device("cpu"),
        "max_instances": 17,
        "instance_selection_seed": 11,
    }
    assert observed["context"]["config"]["wsi_training"]["max_instances"] == 17


def test_bracs_wsi_is_rejected_when_only_annotated_rois_are_available(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Annotated ROI crops cannot stand in for slide-label-only WSI bags."""
    monkeypatch.setattr(
        dataset_adapters.bracs,
        "load_roi_metadata",
        lambda *_: pd.DataFrame(),
    )

    with pytest.raises(ValueError, match="annotated ROI"):
        dataset_adapters._build_bracs(
            {"dataset": {"root": str(tmp_path), "regime": "wsi"}}
        )


def test_camelyon16_wsi_rows_do_not_require_or_read_a_mask(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """CAMELYON16 WSI bags use slide labels and eligible patches, not masks."""
    image = tmp_path / "patch.jpg"
    image.touch()
    monkeypatch.setattr(camelyon16, "list_slide_patches", lambda *_: [(0, image)])
    monkeypatch.setattr(
        camelyon16,
        "load_mask",
        lambda *_: (_ for _ in ()).throw(AssertionError("WSI must not read a mask")),
    )

    rows = dataset_adapters._camelyon16_slide_rows(
        tmp_path, "tumor_001", "tumor", 1, include_patch_labels=False
    )

    assert rows["slide_label"].tolist() == ["tumor"]
    assert "patch_label" not in rows


def test_tuning_uses_the_frozen_initialization_seeds() -> None:
    """Post-freeze CLI seeds must not alter the two locked tuning repetitions."""
    freeze = {
        "seed_roles": {
            "tuning_initialization_0": 101,
            "tuning_initialization_1": 202,
        }
    }

    assert tuning._tuning_seeds(freeze) == [101, 202]


def test_tuning_uses_the_frozen_candidate_grid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A source-code grid change after freeze cannot affect configuration selection."""
    observed: list[dict[str, Any]] = []
    regime = Regime(
        torch.device("cpu"),
        {},
        2,
        False,
        method_grids={"ce": [{"lr": 0.123}]},
    )
    scope = TuningScope(regime, object(), object())

    def evaluate(
        _: str, cfg: dict[str, Any], *__: object
    ) -> tuple[dict[str, Any], dict[str, float]]:
        observed.append(cfg)
        return {}, {"balanced_accuracy": 0.5, "macro_f1": 0.5, "nll": 0.5}

    monkeypatch.setattr(
        "imbalance_benchmark.modeling.workflows.tuning_aggregate._evaluate", evaluate
    )

    _select_trainable("ce", [scope], [7])

    assert observed == [{"lr": 0.123}]


def test_natural_anchor_records_its_support_and_contribution_statistics(
    tmp_path: Path,
) -> None:
    """The descriptive anchor must remain fully auditable alongside controls."""
    rows = pd.DataFrame(
        [
            {"case_id": "p1", "slide_id": "s1", "cancer_type": "A"},
            {"case_id": "p1", "slide_id": "s1", "cancer_type": "A"},
            {"case_id": "p2", "slide_id": "s2", "cancer_type": "B"},
        ]
    )

    natural = write_natural_condition(rows, tmp_path, is_mil=False)

    assert natural["allocated_counts"] == {"A": 2, "B": 1}
    assert natural["achieved_rho"] == 2.0
    assert natural["contribution_stats"]["A"]["pool_fraction_retained"] == 1.0


def test_mil_slide_contribution_uses_one_slide_one_example() -> None:
    """Patch-row multiplicity must not inflate a WSI slide's contribution."""
    rows = pd.DataFrame(
        [
            {"case_id": "p1", "slide_id": "s1", "cancer_type": "A"},
            {"case_id": "p1", "slide_id": "s1", "cancer_type": "A"},
            {"case_id": "p2", "slide_id": "s2", "cancer_type": "A"},
        ]
    )

    stats = contribution_stats(rows, rows, is_mil=True)

    assert stats["A"]["max_slide_contribution"] == 0.5


def test_post_hoc_cost_has_no_trainable_network_parameters() -> None:
    """A reused checkpoint is evaluated but never updated by post-hoc adjustment."""
    cost = cost_payload(
        "post_hoc_logit_adjustment",
        3,
        0.0,
        torch.nn.Linear(2, 2),
        4,
        4,
        0,
    )

    assert cost["trainable_parameters"] == 0


def test_rankmix_cost_records_teacher_and_student_training_footprint() -> None:
    """RankMix trains two networks, not only the final student checkpoint."""
    cost = cost_payload(
        "rankmix",
        3,
        1.0,
        torch.nn.Linear(2, 2),
        4,
        4,
        8,
        training_footprint_parameters=12,
    )

    assert cost["training_footprint_parameters"] == 12


def test_tuning_cost_summarizes_parameter_counts_and_effective_passes() -> None:
    """Validation-search cost must expose its model size and realized data passes."""
    cost = summarize_tuning_cost(
        [
            {
                "processed_examples": 12,
                "unique_training_examples": 6,
                "total_parameters": 10,
                "trainable_parameters": 8,
                "training_footprint_parameters": 20,
            }
        ]
    )

    assert cost["effective_passes_through_unique_examples"] == 2.0
    assert cost["maximum_total_parameters"] == 10
    assert cost["maximum_trainable_parameters"] == 8
    assert cost["maximum_training_footprint_parameters"] == 20
