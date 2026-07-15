from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from imbalance_benchmark.analysis.predictors.rq3_analysis import _covariates
from imbalance_benchmark.datasets.data import BagFeatureDataset
from imbalance_benchmark.construction import locked_class_names
from imbalance_benchmark.manifest.pilot import run_pilot_seed
from imbalance_benchmark.modeling.context import set_training_mode
from imbalance_benchmark.modeling.models import AttentionMil, MLP
from imbalance_benchmark.modeling.workflows.multistage import (
    _freeze_and_reinit_classifier,
)


def test_pilot_construction_and_initialization_seeds_are_separate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    observed: list[int] = []

    monkeypatch.setattr(
        "imbalance_benchmark.manifest.pilot.build_patch_pilot_manifest",
        lambda *_: pd.DataFrame(),
    )
    monkeypatch.setattr(
        "imbalance_benchmark.manifest.pilot.evaluate_pilot_candidate",
        lambda *args, **kwargs: (
            observed.append(kwargs["initialization_seed"]) or (0.5, [0.5])
        ),
    )

    construction_seed = 17
    run_pilot_seed(
        pd.DataFrame(),
        ["A"],
        [5],
        construction_seed,
        object(),
        torch.device("cpu"),
        1,
        False,
        tmp_path,
        1,
        initialization_seed=101,
    )

    assert observed == [101]


def test_native_tail_order_uses_bracs_clinical_label_order() -> None:
    rows = [
        {"cancer_type": label, "split": split}
        for split in ("train", "validation", "test")
        for label in ("IC", "N", "ADH", "PB", "DCIS", "UDH", "FEA")
    ]

    assert locked_class_names(pd.DataFrame(rows)) == [
        "N",
        "PB",
        "UDH",
        "FEA",
        "ADH",
        "DCIS",
        "IC",
    ]


def test_bag_instance_cap_uses_the_frozen_selection_seed(tmp_path: Path) -> None:
    feature = tmp_path / "slide.pt"
    torch.save(torch.arange(24, dtype=torch.float32).reshape(12, 2), feature)
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {
                "case_id": "case",
                "slide_id": "slide",
                "cancer_type": "A",
                "feature_path": feature,
            }
        ]
    ).to_csv(manifest, index=False)

    first, _ = BagFeatureDataset(manifest, max_instances=4, instance_selection_seed=1)[
        0
    ]
    repeat, _ = BagFeatureDataset(manifest, max_instances=4, instance_selection_seed=1)[
        0
    ]
    second, _ = BagFeatureDataset(manifest, max_instances=4, instance_selection_seed=2)[
        0
    ]

    assert torch.equal(first, repeat)
    assert not torch.equal(first, second)


@pytest.mark.parametrize("is_mil", [False, True])
def test_crt_freezes_the_full_representation_in_deterministic_eval_mode(
    is_mil: bool,
) -> None:
    model = (
        AttentionMil(input_dim=4, hidden_dim=3, output_dim=2, dropout=0.5)
        if is_mil
        else MLP(input_dim=4, hidden_dim=3, output_dim=2, dropout=0.5)
    )

    frozen = _freeze_and_reinit_classifier(model, is_mil)
    set_training_mode({"model": model, "frozen_eval_modules": frozen})

    assert all(
        not parameter.requires_grad
        for module in frozen
        for parameter in module.parameters()
    )
    assert all(not module.training for module in frozen)
    assert (
        all(parameter.requires_grad for parameter in model.classifier.parameters())
        if is_mil
        else all(parameter.requires_grad for parameter in model.net[-1].parameters())
    )


def test_rq3_icc_margin_uses_the_fixed_intrinsic_reference(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ref_x = np.array([[1.0], [2.0]])
    ref_y = np.array([0, 1])
    cond_x = np.array([[10.0], [20.0]])
    cond_y = np.array([0, 1])
    seen: dict[str, np.ndarray] = {}

    def feature_frame(path: Path, *_: object) -> tuple[np.ndarray, np.ndarray]:
        if path.name == "manifest_balanced.csv":
            return ref_x, ref_y
        if path.name == "condition.csv":
            return cond_x, cond_y
        return ref_x, ref_y

    monkeypatch.setattr(
        "imbalance_benchmark.analysis.predictors.rq3_analysis._feature_frame",
        feature_frame,
    )
    monkeypatch.setattr(
        "imbalance_benchmark.analysis.predictors.rq3_analysis._feature_identity",
        lambda path, *_: pd.DataFrame(
            {"case_id": ["a", "b"], "slide_id": ["s1", "s2"]}
        ),
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

    def margins(x: np.ndarray, *_: object) -> np.ndarray:
        seen["x"] = x
        return np.array([0.1, 0.2])

    monkeypatch.setattr(
        "imbalance_benchmark.analysis.predictors.rq3_analysis.class_margin_cross_fit",
        margins,
    )

    condition_path = tmp_path / "condition.csv"
    condition_path.touch()
    _covariates(
        {"data": tmp_path},
        False,
        {"path": str(condition_path), "contribution_stats": {}},
    )

    assert np.array_equal(seen["x"], ref_x)
